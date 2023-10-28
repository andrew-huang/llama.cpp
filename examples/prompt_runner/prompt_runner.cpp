// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"
#include "build-info.h"

#include "json.hpp"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>

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

#include "grammar-parser.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

using json = nlohmann::json;

static llama_context           ** g_ctx;
static llama_model             ** g_model;

static size_t   benchmark_start_time;

const char* ws = "\n\r";

// trim from end of string (right)
inline std::string& rtrim_nl(std::string& s, const char* t = ws)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim_nl(std::string& s, const char* t = ws)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim_nl(std::string& s, const char* t = ws)
{
    return ltrim_nl(rtrim_nl(s, t), t);
}

static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
    // if first bit is 1, meaning it's a partial character
    if (out.size() > 0 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "\\x" + res;
    }
    return out;
}

static std::string now_timestr() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
    auto str = oss.str();
    return str;
}

json make_token_respose(std::vector<std::string> &responses, int cur_test_nr,
    int total_tests, const std::string &test_id, json tokens, size_t prompt_token_cnt, json expected);

json make_token_respose(std::vector<std::string> &responses, int cur_test_nr,
    int total_tests, const std::string &test_id, json tokens, size_t prompt_token_cnt, json expected)
{
    if (cur_test_nr <= 0)
        cur_test_nr = 1;

    int passed_time = time(NULL) - benchmark_start_time;
    float time_per_test = ((float) passed_time) / (float) cur_test_nr;
    float remaining = time_per_test * (float) (total_tests - cur_test_nr);
    remaining /= 60.0;

    float passed_time_mins = ((float) passed_time) / 60.0;

    std::string info_str =
        "[" + std::to_string(cur_test_nr) + "/" + std::to_string(total_tests)
        + "| id=" + test_id + " #p=" + std::to_string((int) prompt_token_cnt) + "]: " + tokens.dump(-1);
    printf(
        "[s/t=%5.2fs [eta=%5.1fm, t=%5.1fm]] %s\n",
        time_per_test, remaining, passed_time_mins, info_str.c_str());
    fflush(stdout);

    responses.push_back(test_id + "=" + info_str);

    json single_response;
    single_response["test_id"] = test_id;
    single_response["tokens"] = tokens;
    single_response["expected"] = expected;
    single_response["prompt_token_count"] = (int) prompt_token_cnt;
    single_response["timestamp"] = (int) time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

json make_response(std::vector<std::string> &responses, int cur_test_nr,
    int total_tests, const std::string &test_id, float temp,
    int64_t seed, const std::vector<llama_token> &embd_gen, size_t prompt_token_cnt, json expected);

json make_response(std::vector<std::string> &responses, int cur_test_nr,
    int total_tests, const std::string &test_id, float temp,
    int64_t seed, const std::vector<llama_token> &embd_gen, size_t prompt_token_cnt, json expected)
{
    llama_context *ctx = *g_ctx;

    std::ostringstream oss;
    oss << std::setprecision(1) << temp;
    std::string temp_str = oss.str();

    if (cur_test_nr <= 0)
        cur_test_nr = 1;

    int passed_time = time(NULL) - benchmark_start_time;
    float time_per_test = ((float) passed_time) / (float) cur_test_nr;
    float remaining = time_per_test * (float) (total_tests - cur_test_nr);
    remaining /= 60.0;

    float passed_time_mins = ((float) passed_time) / 60.0;

    std::string gen_prefix =
        "[" + std::to_string(cur_test_nr) + "/" + std::to_string(total_tests)
        + "| id=" + test_id
        + ", temp=" + temp_str
        + ", seed=" + std::to_string(seed)
        + ", #p=" + std::to_string((int) prompt_token_cnt)
        + ", #g=" + std::to_string((int) embd_gen.size())
        + "]:";

    std::string gen = "";
    for (auto id : embd_gen) {
        gen += tokens_to_output_formatted_string(ctx, id);
    }

    printf("[s/t=%5.2fs [eta=%5.1fm, t=%5.1fm]] %s %s\n",
        time_per_test, remaining, passed_time_mins, gen_prefix.c_str(), gen.c_str());
    fflush(stdout);

    responses.push_back(gen_prefix + gen);

    json single_response;
    single_response["test_id"] = test_id;
    single_response["seed"] = seed;
    single_response["temp"] = temp_str;
    single_response["response"] = gen;
    single_response["expected"] = expected;
    single_response["prompt_token_count"] = (int) prompt_token_cnt;
    single_response["generated_token_count"] = (int) embd_gen.size();
    single_response["timestamp"] = (int) time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

int process_prompt(const std::string &prompt, const gpt_params &params);

int process_prompt(const std::string &prompt, const gpt_params &params) {
    llama_context *ctx = *g_ctx;

    const bool add_bos = llama_vocab_type(*g_model) == LLAMA_VOCAB_TYPE_SPM;

    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx, prompt.c_str(), add_bos, true);

    int n_past = 0;

    for (int i = 0; i < (int) embd_inp.size(); i += params.n_batch) {
        int n_eval = (int) embd_inp.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }
        //d// printf("### PROC: i=%d, n_eval=%d, n_past=%d\n", i, n_eval, n_past);
        fflush(stdout);
        if (llama_decode(ctx, llama_batch_get_one(&embd_inp[i], n_eval, n_past, 0))) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return -1;
        }

        n_past += n_eval;
    }

    return n_past;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    llama_sampling_params & sparams = params.sparams;

    if (!sparams.grammar.empty()) {
        grammar_parser::parse_state parsed_grammar;
        parsed_grammar = grammar_parser::parse(sparams.grammar.c_str());

        fprintf(stderr, "%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    std::string filename = "prompt_runner_config.json";
    std::ifstream file(filename);
    if (!file) {
        fprintf(stderr, "error: failed to open prompt_runner_config config file '%s'\n", filename.c_str());
        return 1;
    }

    const json prompt_runner_conf = json::parse(file);

    if (prompt_runner_conf.find("prompt_tests") == prompt_runner_conf.end()) {
        fprintf(stderr, "**********\n");
        fprintf(stderr, "ERROR: No prompt_tests in prompt_runner_config.json!\n");
        fprintf(stderr, "**********\n");

        return 1;
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
    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }
    fprintf(stderr, "loaded model\n");
    fflush(stderr);

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    if (params.n_ctx < 8) {
        fprintf(stderr, "%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }


    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // Add BOS if SPM tokenizer
    const bool add_bos = llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM;

    // tokenize the prompt
    std::vector<llama_token> embd_inp;
    std::vector<std::string> responses;
    json j_resps;
    json j_tok_resps;

    bool first = true;

    bool record_next_token_info = prompt_runner_conf.value("record_next_token_info", false);

    std::vector<int64_t> sample_seeds;
    if (prompt_runner_conf.find("sample_seeds") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading sample_seeds\n");
        for (const auto &seed_value : prompt_runner_conf["sample_seeds"]) {
            int64_t seed = seed_value;
            sample_seeds.push_back(seed);
        }
    }

    std::vector<float> temps;

    if (prompt_runner_conf.find("temps") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading temps\n");
        for (const auto &temp : prompt_runner_conf["temps"]) {
            temps.push_back(temp);
        }
    } else {
        temps.push_back(sparams.temp);
    }

    std::vector<int64_t> seeds;
    if (prompt_runner_conf.find("seeds") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading seeds\n");
        for (const auto &seed_value : prompt_runner_conf["seeds"]) {
            int64_t seed = seed_value;
            printf("seed value: %ld\n", seed);
            seeds.push_back(seed);
        }
    } else {
        seeds.push_back(params.seed);
    }

    int test_count = 0;
    if (prompt_runner_conf.find("prompt_tests") != prompt_runner_conf.end()) {
        test_count = prompt_runner_conf["prompt_tests"].size();
    } else {
        fprintf(stderr, "No \"prompt_tests\" defined in the prompt_runner_config.json!\n");
        return 1;
    }

    int total_tests = seeds.size() * temps.size() * test_count;
    int current_test_nr = 0;

    benchmark_start_time = time(NULL);

    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        std::string prompt = params.prompt;

        json expected;
        if (prompt_test.find("expected") != prompt_test.end()) {
            expected = prompt_test["expected"];
        }

        std::string test_id = prompt_test.value("id", "unknown_test_id");

        if (prompt_runner_conf.find("replacements") != prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                prompt = std::regex_replace(prompt, std::regex(repl[0]), replacement);
            }
        }

        std::string repl_info = "";

        if (prompt_test.find("replacements") != prompt_test.end()) {
            for (const auto &repl : prompt_test["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                if (replacement.size() < 250) {
                    repl_info += search + " := " +  replacement + "\n";
                }
                prompt = std::regex_replace(prompt, std::regex(repl[0]), replacement);
            }
        }

        rtrim_nl(prompt);

        if (params.verbose_prompt) {
            printf("PROMPT------------------------\n%s\n-----------------------------\n", prompt.c_str());
        }

        printf("------------------------\n%s", repl_info.c_str());
        fflush(stdout);

        fprintf(stderr, "PROMPT-RUNNER-START\n");
        fflush(stderr);

        for (auto &temp : temps) {
            sparams.temp = temp;

            for (const auto &seed_value : seeds) {
                int64_t seed = seed_value;
                llama_set_rng_seed(ctx, seed);

                struct llama_sampling_context * ctx_sampling = llama_sampling_init(sparams);

                current_test_nr += 1;

                if (params.prompt.empty()) {
                    fprintf(stderr, "No prompt given!");
                    return 1;
                }

                embd_inp = ::llama_tokenize(ctx, prompt.c_str(), add_bos, true);

                const int n_ctx = llama_n_ctx(ctx);
                const int embd_inp_prompt_size = embd_inp.size();

                if (((int) embd_inp.size() + (int) params.n_predict) > n_ctx - 4) {
                    fprintf(stderr, "%s: error: prompt is too long (%d tokens, %d predict, max %d)\n", __func__, (int) embd_inp.size(), (int) params.n_predict, n_ctx - 4);
                    return 1;
                }

                // number of tokens to keep when resetting context
                if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
                    params.n_keep = (int)embd_inp.size();
                }

                if (params.verbose_prompt) {
                    fprintf(stderr, "\n");
                    fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
                    fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
                    for (int i = 0; i < (int) embd_inp.size(); i++) {
                        fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
                    }

                    if (params.n_keep > 0) {
                    fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
                        for (int i = 0; i < params.n_keep; i++) {
                            fprintf(stderr, "%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
                        }
                        fprintf(stderr, "'\n");
                    }
                    fprintf(stderr, "\n");
                }

                if (first) {
                    fprintf(stderr, "sampling: \n%s\n", llama_sampling_print(sparams).c_str());
                }

                int n_past             = 0;
                int n_remain           = params.n_predict;
                int n_consumed         = 0;

                llama_kv_cache_tokens_rm(ctx, -1, -1);

                std::vector<llama_token> embd;
                std::vector<llama_token> embd_gen;

                while (n_remain != 0) {
                    // predict
                    if (embd.size() > 0) {
                        // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
                        // --prompt or --file which uses the same value.
                        auto max_embd_size = n_ctx - 4;
                        // Ensure the input doesn't exceed the context size by truncating embd if necessary.
                        if ((int)embd.size() > max_embd_size) {
                            auto skipped_tokens = embd.size() - max_embd_size;
                            printf("<<input too long: skipped %zu token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                            fflush(stdout);
                            embd.resize(max_embd_size);
                        }

                        // evaluate tokens in batches
                        // embd is typically prepared beforehand to fit within a batch, but not always
                        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                            int n_eval = (int) embd.size() - i;
                            if (n_eval > params.n_batch) {
                                n_eval = params.n_batch;
                            }
                            if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                                fprintf(stderr, "%s : failed to eval\n", __func__);
                                return 1;
                            }
                            n_past += n_eval;
                        }
                    }

                    embd.clear();

                    if ((int) embd_inp.size() <= n_consumed) {
                        if (record_next_token_info) {
                            const int n_vocab = llama_n_vocab(model);

                            float * logits = llama_get_logits_ith(ctx, 0);

                            std::vector<llama_token_data> cur;
                            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                                cur.emplace_back(
                                    llama_token_data{token_id, logits[token_id], 0.0f});
                            }

                            llama_token_data_array candidates_p =
                                { cur.data(), cur.size(), false };
                            llama_sample_softmax(nullptr, &candidates_p);

                            json tokens;

                            for (size_t i = 0; i < candidates_p.size; ++i) {
                                if (candidates_p.data[i].p > 0.00001) {
                                    std::string tok =
                                        tokens_to_output_formatted_string(
                                            ctx, candidates_p.data[i].id);
                                    json j_tok;
                                    j_tok[0] = tok;
                                    j_tok[1] = candidates_p.data[i].p;
                                    tokens.push_back(j_tok);
                                }
                            }

                            if (tokens.size() > 1) {
                                j_tok_resps.push_back(
                                    make_token_respose(
                                        responses, current_test_nr,
                                        total_tests, test_id, tokens, embd_inp_prompt_size, expected));
                            }
                        }


                        // The sample_seeds mechanism is a cheat, for the case where we just only
                        // need the next token. We just reroll the selected candidate and
                        // record the selections.
                        // XXX: This really only works if you need one response token!
                        int seeds_remaining = 1;
                        int sample_seeds_idx = -1;
                        if (sample_seeds.size() > 0) {
                            sample_seeds_idx = 0;
                            seeds_remaining = sample_seeds.size();
                        }

                        llama_token id = 0;
                        while (seeds_remaining > 0) {
                            if (sample_seeds_idx >= 0) {
                                seed = sample_seeds[sample_seeds_idx];
                                llama_set_rng_seed(ctx, seed);
                                sample_seeds_idx++;
                            }

                            id = llama_sampling_sample(ctx_sampling, ctx, nullptr);

                            if (sample_seeds.size() > 0 && id != llama_token_eos(model)) {
                                std::vector<llama_token> embd_single_id;
                                embd_single_id.push_back(id);
                                j_resps.push_back(make_response(
                                    responses, current_test_nr, total_tests,
                                    test_id, temp, seed, embd_single_id, embd_inp_prompt_size, expected));
                            }

                            seeds_remaining -= 1;
                        }

                        // add it to the context
                        embd.push_back(id);

                        if (sample_seeds.size() == 0) {
                            embd_gen.push_back(id);
                        }

                        // decrement remaining sampling budget
                        --n_remain;
                    } else {
                        while ((int) embd_inp.size() > n_consumed) {
                            embd.push_back(embd_inp[n_consumed]);
                            // push the prompt in the sampling context in order to apply repetition penalties later
                            // for the prompt, we don't apply grammar rules
                            llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);
                            ++n_consumed;
                        }
                    }

                    // end of text token
                    if (!embd.empty() && embd.back() == llama_token_eos(model)) {
                        // fprintf(stderr, " [end of text]\n");
                        break;
                    }

                    if (prompt_runner_conf.find("stop_sequences")
                        != prompt_runner_conf.end())
                    {
                        std::string generated = "";
                        for (auto id : embd_gen) {
                            generated += llama_token_to_piece(ctx, id);
                        }

                        for (const auto &matched : prompt_runner_conf["stop_sequences"])
                        {
                            if (generated.find(matched) != std::string::npos)
                            {
                                n_remain = 0;
                                break;
                            }
                        }
                    }
                }

                if (sample_seeds.size() == 0) {
                    j_resps.push_back(
                        make_response(
                            responses, current_test_nr, total_tests, test_id,
                            sparams.temp, seed, embd_gen, embd_inp_prompt_size, expected));
                }

    //            std::string gen_prefix = "[id=" + test_id + ", seed=" + std::to_string(seed) + "]: ";
    //            std::string gen = "";
    //            for (auto id : embd_gen) {
    //                gen += llama_token_to_piece(ctx, id);
    //            }
    //            printf("%s %s\n", gen_prefix.c_str(), gen.c_str());
    //            fflush(stdout);
    //
    //            responses.push_back(gen_prefix + gen);
    //
    //            json single_response;
    //            single_response["test_id"] = test_id;
    //            single_response["seed"] = seed;
    //            single_response["response"] = gen;
    //            j_resps.push_back(single_response);


                llama_sampling_free(ctx_sampling);

                first = false;

            }
        }
    }

    llama_print_timings(ctx);

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    printf("model: %s\n", model_file.c_str());

    for (auto resp : responses) {
        printf("%s\n", resp.c_str());
    }

    std::string responses_json_dump = j_resps.dump(2);
    printf("%s\n", responses_json_dump.c_str());
    fflush(stdout);

    json j_params;
    j_params["rope_freq_base"] = params.rope_freq_base;
    j_params["rope_freq_scale"] = params.rope_freq_scale;
    j_params["temp"] = sparams.temp;
    j_params["top_k"] = sparams.top_k;
    j_params["top_p"] = sparams.top_p;
    j_params["tfs_z"] = sparams.tfs_z;
    j_params["typical_p"] = sparams.typical_p;
    j_params["repeat_last_n"] = sparams.penalty_last_n;
    j_params["repeat_penality"] = sparams.penalty_repeat;

    json results;
    results["params"] = j_params;
    results["model_file"] = model_file;
    results["prompt"] = std::string(params.prompt);
    results["config"] = prompt_runner_conf;
    results["results"] = j_resps;
    if (record_next_token_info) {
        results["tokens"] = j_tok_resps;
    }

    std::string out_file_name = "result_" + std::to_string(time(NULL)) + "_" + model_file + ".json";
    std::ofstream outf(out_file_name);
    outf << results.dump(2);
    outf.close();

    printf("[PROMPT_RUNNER_OUTPUT_FILE: %s]\n", out_file_name.c_str());
    fflush(stdout);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
