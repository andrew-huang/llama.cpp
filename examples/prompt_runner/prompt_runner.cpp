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

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

using json = nlohmann::json;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        printf("\n");
        _exit(130);
    }
}
#endif

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
    std::string out = token == -1 ? "" : llama_token_to_str(ctx, token);
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

int process_prompt(llama_context *ctx, const std::string &prompt, const gpt_params &params) {
    const int n_ctx = llama_n_ctx(ctx);

    std::vector<llama_token> embd_inp =
        ::llama_tokenize(ctx, prompt.c_str(), true);

    // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
    auto max_embd_size = n_ctx - 4;

    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if ((int)embd_inp.size() > max_embd_size) {
        auto skipped_tokens = embd_inp.size() - max_embd_size;
        printf("<<input too long: skipped %zu token%s>>",
            skipped_tokens, skipped_tokens != 1 ? "s" : "");
        fflush(stdout);
        embd_inp.resize(max_embd_size);
    }

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n",
            __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n",
                embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }

        fprintf(stderr, "\n");
    }

    int n_past = 0;

    int n_eval = (int) embd_inp.size();

    printf("### PROC: n_eval=%d, n_past=%d\n", n_eval, n_past);

    printf("Prompt proc: ");
    for (int d = 0; d < (0 + n_eval); d++) {
        printf("%s", llama_token_to_str(ctx, embd_inp[d]));
    }
    printf("\n");

    if (llama_eval(ctx, &embd_inp[0], n_eval, n_past, params.n_threads)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return -1;
    }

    n_past += n_eval;

    return n_past;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    llama_backend_init(params.numa);

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    // load the model and apply lora adapter, if any
    auto lparams = llama_context_params_from_gpt_params(params);
//    auto lparams = llama_context_default_params();

    // init
    model = llama_load_model_from_file(params.model.c_str(), lparams);
    if (model == nullptr) {
        return 1;
    }

    ctx = llama_new_context_with_model(model, lparams);
    if (ctx == nullptr) {
        llama_free_model(model);
        return 1;
    }
//    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    std::vector<std::string> responses;

    std::string prompt = "The quick brown fox";

    // TODO: replace with ring-buffer
    const int n_ctx = llama_n_ctx(ctx);

    int n_past_cached = process_prompt(ctx, prompt, params);

    // TODO WEICON:
    // - save state to memory here, destroy context

    llama_set_rng_seed(ctx, 32);

    const size_t n_state_size_max = llama_get_state_size(ctx);
    uint8_t *state_mem = new uint8_t[n_state_size_max];
    const size_t n_state_size_cur = llama_copy_state_data(ctx, state_mem);
    printf("n_state_size_cur=%d max=%d\n", (int) n_state_size_cur, (int) n_state_size_max);

    llama_free(ctx);
    ctx = nullptr;

    printf("PROMPT------------------------\n%s\n-----------------------------\n", prompt.c_str());

    {
        llama_context *ctx2 = nullptr;
        ctx2 = llama_new_context_with_model(model, lparams);
        int set_len = llama_set_state_data(ctx2, state_mem);

        delete[] state_mem;

//  n_past_cached = process_prompt(ctx2, prompt, params);

        int n_past = n_past_cached;

        for (int pred = 0; pred < 10; pred++) {
            auto logits  = llama_get_logits(ctx2);
            auto n_vocab = llama_n_vocab(ctx2);
            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
            llama_token next_token = llama_sample_token(ctx2, &candidates_p);
            std::string tok = llama_token_to_str(ctx2, next_token);
            printf("tok=%s\n", tok.c_str());

            if (llama_eval(ctx2, &next_token, 1, n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
            n_past += 1;
        }

        llama_free(ctx2);
        ctx2 = nullptr;
    }


    llama_free_model(model);
    llama_backend_free();

    return 0;
}
