// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

#include "json.hpp"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <regex>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

using json = nlohmann::json;

static console_state con_st;
static llama_context ** g_ctx;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        console_cleanup(con_st);
        printf("\n");
        llama_print_timings(*g_ctx);
        _exit(130);
    }
}
#endif

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    std::string filename = "prompt_runner_config.json";
    std::ifstream file(filename);
    if (!file) {
        fprintf(stderr, "error: failed to open prompt_runner_config config file '%s'\n", filename.c_str());
        return 1;
    }

    const json prompt_runner_conf = json::parse(file);

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_st.use_color = params.use_color;
    con_st.multiline_input = params.multiline_input;
    console_init(con_st);
    atexit([]() { console_cleanup(con_st); });

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 10000.0) {
        fprintf(stderr, "%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        fprintf(stderr, "%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx > 2048) {
        // TODO: determine the actual max context of the model (e.g. 4096 for LLaMA v2) and use that instead of 2048
        fprintf(stderr, "%s: warning: base model only supports context sizes no greater than 2048 tokens (%d specified)\n", __func__, params.n_ctx);
    } else if (params.n_ctx < 8) {
        fprintf(stderr, "%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;
    llama_context * ctx_guidance = NULL;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (params.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_ctx parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            fprintf(stderr, "%s: testing memory usage for n_batch = %d, n_ctx = %d\n", __func__, params.n_batch, params.n_ctx);

            const std::vector<llama_token> tmp(params.n_batch, llama_token_bos());
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_ctx, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);
        llama_free_model(model);

        return 0;
    }

    // export the cgraph and exit
    if (params.export_cgraph) {
        llama_eval_export(ctx, "llama.ggml");
        llama_free(ctx);
        llama_free_model(model);

        return 0;
    }

    // tokenize the prompt
    std::vector<llama_token> embd_inp;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // do one empty run to warm up the model
    {
        const std::vector<llama_token> tmp = { llama_token_bos(), };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(ctx);
    }

    std::vector<std::string> responses;
    json j_resps;

    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        std::string prompt = params.prompt;

        std::string test_id = prompt_test["id"];

        for (const auto &repl : prompt_test["replacements"]) {
            std::string search = repl[0];
            std::string replacement = repl[1];
            prompt = std::regex_replace(prompt, std::regex(repl[0]), replacement);
        }

        for (const auto &seed_value : prompt_runner_conf["seeds"]) {
            int64_t seed = seed_value;

            printf("SEED=%d\n", (int) seed);
            llama_set_rng_seed(ctx, seed);

            if (params.prompt.empty()) {
                fprintf(stderr, "No prompt given!");
                return 1;
            }

            embd_inp = ::llama_tokenize(ctx, prompt.c_str(), true);

            // Tokenize negative prompt
            std::vector<llama_token> guidance_inp;
            int guidance_offset = 0;
            int original_prompt_len = 0;
            if (ctx_guidance) {
                params.cfg_negative_prompt.insert(0, 1, ' ');
                guidance_inp = ::llama_tokenize(ctx_guidance, params.cfg_negative_prompt, true);

                std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, true);
                original_prompt_len = original_inp.size();
                guidance_offset = (int)guidance_inp.size() - original_prompt_len;
            }

            const int n_ctx = llama_n_ctx(ctx);

            if ((int) embd_inp.size() > n_ctx - 4) {
                fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
                return 1;
            }

            // number of tokens to keep when resetting context
            if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
                params.n_keep = (int)embd_inp.size();
            }

            // determine newline token
            auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

            if (params.verbose_prompt) {
                fprintf(stderr, "\n");
                fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
                fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
                for (int i = 0; i < (int) embd_inp.size(); i++) {
                    fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
                }

                if (ctx_guidance) {
                    fprintf(stderr, "\n");
                    fprintf(stderr, "%s: negative prompt: '%s'\n", __func__, params.cfg_negative_prompt.c_str());
                    fprintf(stderr, "%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
                    for (int i = 0; i < (int) guidance_inp.size(); i++) {
                        fprintf(stderr, "%6d -> '%s'\n", guidance_inp[i], llama_token_to_str(ctx, guidance_inp[i]));
                    }
                }

                if (params.n_keep > 0) {
                fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
                    for (int i = 0; i < params.n_keep; i++) {
                        fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
                    }
                    fprintf(stderr, "'\n");
                }
                fprintf(stderr, "\n");
            }

            fprintf(stderr, "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
                    params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
            fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
            fprintf(stderr, "\n\n");

            grammar_parser::parse_state parsed_grammar;
            llama_grammar *             grammar = NULL;
            if (!params.grammar.empty()) {
                parsed_grammar = grammar_parser::parse(params.grammar.c_str());
                // will be empty (default) if there are parse errors
                if (parsed_grammar.rules.empty()) {
                    return 1;
                }
                fprintf(stderr, "%s: grammar:\n", __func__);
                grammar_parser::print_grammar(stderr, parsed_grammar);
                fprintf(stderr, "\n");

                {
                    auto it = params.logit_bias.find(llama_token_eos());
                    if (it != params.logit_bias.end() && it->second == -INFINITY) {
                        fprintf(stderr,
                            "%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
                    }
                }

                std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
                grammar = llama_grammar_init(
                    grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
            }

            // TODO: replace with ring-buffer
            std::vector<llama_token> last_n_tokens(n_ctx);
            std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

            bool is_antiprompt        = false;
            bool input_echo           = true;

            int n_past             = 0;
            int n_remain           = params.n_predict;
            int n_consumed         = 0;
            int n_past_guidance    = 0;

            // the first thing we will do is to output the prompt, so set color accordingly
            console_set_color(con_st, CONSOLE_COLOR_PROMPT);

            std::vector<llama_token> embd;
            std::vector<llama_token> embd_gen;
            std::vector<llama_token> embd_guidance;

            while ((n_remain != 0 && !is_antiprompt)) {
                // predict
                if (embd.size() > 0) {
                    // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
                    // --prompt or --file which uses the same value.
                    auto max_embd_size = n_ctx - 4;
                    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
                    if ((int)embd.size() > max_embd_size) {
                        auto skipped_tokens = embd.size() - max_embd_size;
                        console_set_color(con_st, CONSOLE_COLOR_ERROR);
                        printf("<<input too long: skipped %zu token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                        console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
                        fflush(stdout);
                        embd.resize(max_embd_size);
                    }

                    // infinite text generation via context swapping
                    // if we run out of context:
                    // - take the n_keep first tokens from the original prompt (via n_past)
                    // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                    if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {
                        const int n_left = n_past - params.n_keep;

                        // always keep the first token - BOS
                        n_past = std::max(1, params.n_keep);
                        n_past_guidance = std::max(1, params.n_keep + guidance_offset);

                        // insert n_left/2 tokens at the start of embd from last_n_tokens
                        embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                        //printf("\n---\n");
                        //printf("resetting: '");
                        //for (int i = 0; i < (int) embd.size(); i++) {
                        //    printf("%s", llama_token_to_str(ctx, embd[i]));
                        //}
                        //printf("'\n");
                        //printf("\n---\n");
                    }

                    // evaluate tokens in batches
                    // embd is typically prepared beforehand to fit within a batch, but not always

                    if (ctx_guidance) {
                        int input_size = 0;
                        llama_token* input_buf = NULL;

                        if (n_past_guidance < (int) guidance_inp.size()) {
                            // Guidance context should have the same data with these modifications:
                            //
                            // * Replace the initial prompt
                            // * Shift everything by guidance_offset
                            embd_guidance = guidance_inp;
                            if (embd.begin() + original_prompt_len < embd.end()) {
                                embd_guidance.insert(
                                    embd_guidance.end(),
                                    embd.begin() + original_prompt_len,
                                    embd.end()
                                );
                            }

                            input_buf = embd_guidance.data();
                            input_size = embd_guidance.size();
                            //fprintf(stderr, "\n---------------------\n");
                            //for (int i = 0; i < (int) embd_guidance.size(); i++) {
                                //fprintf(stderr, "%s", llama_token_to_str(ctx, embd_guidance[i]));
                            //}
                            //fprintf(stderr, "\n---------------------\n");
                        } else {
                            input_buf = embd.data();
                            input_size = embd.size();
                        }

                        for (int i = 0; i < input_size; i += params.n_batch) {
                            int n_eval = std::min(input_size - i, params.n_batch);
                            if (llama_eval(ctx_guidance, input_buf + i, n_eval, n_past_guidance, params.n_threads)) {
                                fprintf(stderr, "%s : failed to eval\n", __func__);
                                return 1;
                            }

                            n_past_guidance += n_eval;
                        }
                    }

                    for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                        int n_eval = (int) embd.size() - i;
                        if (n_eval > params.n_batch) {
                            n_eval = params.n_batch;
                        }
                        if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                            fprintf(stderr, "%s : failed to eval\n", __func__);
                            return 1;
                        }
                        n_past += n_eval;
                    }
                }

                embd.clear();
                embd_guidance.clear();

                if ((int) embd_inp.size() <= n_consumed) {
                    // out of user input, sample next token
                    const float   temp            = params.temp;
                    const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
                    const float   top_p           = params.top_p;
                    const float   tfs_z           = params.tfs_z;
                    const float   typical_p       = params.typical_p;
                    const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
                    const float   repeat_penalty  = params.repeat_penalty;
                    const float   alpha_presence  = params.presence_penalty;
                    const float   alpha_frequency = params.frequency_penalty;
                    const int     mirostat        = params.mirostat;
                    const float   mirostat_tau    = params.mirostat_tau;
                    const float   mirostat_eta    = params.mirostat_eta;
                    const bool    penalize_nl     = params.penalize_nl;

                    llama_token id = 0;

                    {
                        auto logits  = llama_get_logits(ctx);
                        auto n_vocab = llama_n_vocab(ctx);

                        // Apply params.logit_bias map
                        for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                            logits[it->first] += it->second;
                        }

                        std::vector<llama_token_data> candidates;
                        candidates.reserve(n_vocab);
                        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                        }

                        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                        if (ctx_guidance) {
                            llama_sample_classifier_free_guidance(ctx, &candidates_p, ctx_guidance, params.cfg_scale);
                        }

                        // Apply penalties
                        float nl_logit = logits[llama_token_nl()];
                        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                        llama_sample_repetition_penalty(ctx, &candidates_p,
                            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                            last_n_repeat, repeat_penalty);
                        llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                            last_n_repeat, alpha_frequency, alpha_presence);
                        if (!penalize_nl) {
                            logits[llama_token_nl()] = nl_logit;
                        }

                        if (grammar != NULL) {
                            llama_sample_grammar(ctx, &candidates_p, grammar);
                        }

                        if (temp <= 0) {
                            // Greedy sampling
                            id = llama_sample_token_greedy(ctx, &candidates_p);
                        } else {
                            if (mirostat == 1) {
                                static float mirostat_mu = 2.0f * mirostat_tau;
                                const int mirostat_m = 100;
                                llama_sample_temperature(ctx, &candidates_p, temp);
                                id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                            } else if (mirostat == 2) {
                                static float mirostat_mu = 2.0f * mirostat_tau;
                                llama_sample_temperature(ctx, &candidates_p, temp);
                                id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                            } else {
                                // Temperature sampling
                                llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                                llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                                llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                                llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                                llama_sample_temperature(ctx, &candidates_p, temp);
                                id = llama_sample_token(ctx, &candidates_p);
                            }
                        }
                        // printf("`%d`", candidates_p.size);

                        if (grammar != NULL) {
                            llama_grammar_accept_token(ctx, grammar, id);
                        }

                        last_n_tokens.erase(last_n_tokens.begin());
                        last_n_tokens.push_back(id);
                    }

                    // add it to the context
                    embd.push_back(id);
                    embd_gen.push_back(id);

                    // echo this to console
                    input_echo = true;

                    // decrement remaining sampling budget
                    --n_remain;
                } else {
                    // some user input remains from prompt or interaction, forward it to processing
                    while ((int) embd_inp.size() > n_consumed) {
                        embd.push_back(embd_inp[n_consumed]);
                        last_n_tokens.erase(last_n_tokens.begin());
                        last_n_tokens.push_back(embd_inp[n_consumed]);
                        ++n_consumed;
                        if ((int) embd.size() >= params.n_batch) {
                            break;
                        }
                    }
                }

                // display text
                if (input_echo) {
                    for (auto id : embd) {
                        printf("%s", llama_token_to_str(ctx, id));
                    }
                    fflush(stdout);
                }
                // reset color to default if we there is no pending user input
                if (input_echo && (int)embd_inp.size() == n_consumed) {
                    console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
                }

                // end of text token
                if (!embd.empty() && embd.back() == llama_token_eos()) {
                    fprintf(stderr, " [end of text]\n");
                    break;
                }
            }


            std::string gen_prefix = "[id=" + test_id + ", seed=" + std::to_string(seed) + "]: ";
            std::string gen = "";
            for (auto id : embd_gen) {
                gen += llama_token_to_str(ctx, id);
            }
            responses.push_back(gen_prefix + gen);

            json single_response;
            single_response["test_id"] = test_id;
            single_response["seed"] = seed;
            single_response["response"] = gen;
            j_resps.push_back(single_response);

            llama_print_timings(ctx);

            if (grammar != NULL) {
                llama_grammar_free(grammar);
            }
        }
    }
    std::string model_file = params.model.c_str();
    printf("model: %s\n", model_file.c_str());

    for (auto resp : responses) {
        printf("%s\n", resp.c_str());
    }

    std::string responses_json_dump = j_resps.dump(2);
    printf("%s\n", responses_json_dump.c_str());
    json results;
    results["model_file"] = model_file;
    results["prompt"] = std::string(params.prompt);
    results["config"] = prompt_runner_conf;
    results["results"] = j_resps;

    std::string out_file_name = "result_" + std::to_string(time(NULL)) + ".json";
    std::ofstream outf(out_file_name);
    outf << results.dump(2);
    outf.close();

    if (ctx_guidance) { llama_free(ctx_guidance); }
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
