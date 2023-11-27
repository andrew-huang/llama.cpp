// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "json.hpp"
#include "llama.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <signal.h>
#include <windows.h>
#endif

#include "grammar-parser.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

using json = nlohmann::json;

static llama_context **g_ctx;
static llama_model **g_model;

static size_t benchmark_start_time;

const char *ws = "\n\r";

std::string concatl(const std::vector<std::string> &l, bool indent = false);

std::string concatl(const std::vector<std::string> &l, bool indent) {
    std::string logstr;
    for (auto logentry : l) {
        if (indent) {
            std::string le = logentry;
            std::replace(le.begin(), le.end(), '\n', ' ');
            logstr += "    " + le + "\n\n";
        } else {
            logstr += logentry;
        }
    }
    return logstr;
}

// trim from end of string (right)
inline std::string &rtrim_nl(std::string &s, const char *t = ws) {
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string &ltrim_nl(std::string &s, const char *t = ws) {
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string &trim_nl(std::string &s, const char *t = ws) {
    return ltrim_nl(rtrim_nl(s, t), t);
}

static std::string tokens_to_output_formatted_string(const llama_context *ctx,
                                                     const llama_token token) {
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
    // if first bit is 1, meaning it's a partial character
    if (out.size() > 0 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "\\x" + res;
    }
    return out;
}

json sparams_to_json(llama_sampling_params &sp);
json sparams_to_json(llama_sampling_params &sp) {
    // clang-format off
    //    int32_t n_prev            = 64;    // number of previous tokens to remember
    //    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
    //    int32_t top_k             = 40;    // <= 0 to use vocab size
    //    float   top_p             = 0.95f; // 1.0 = disabled
    //    float   tfs_z             = 1.00f; // 1.0 = disabled
    //    float   typical_p         = 1.00f; // 1.0 = disabled
    //    float   temp              = 0.80f; // 1.0 = disabled
    //    int32_t penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    //    float   penalty_repeat    = 1.10f; // 1.0 = disabled
    //    float   penalty_freq      = 0.00f; // 0.0 = disabled
    //    float   penalty_present   = 0.00f; // 0.0 = disabled
    //    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    //    float   mirostat_tau      = 5.00f; // target entropy
    //    float   mirostat_eta      = 0.10f; // learning rate
    //    bool    penalize_nl       = true;  // consider newlines as a repeatable token
    // clang-format on

    json j_params;
    j_params["top_k"] = sp.top_k;
    j_params["top_p"] = sp.top_p;
    j_params["min_p"] = sp.min_p;
    j_params["tfs_z"] = sp.tfs_z;
    j_params["typical_p"] = sp.typical_p;
    j_params["temp"] = sp.temp;
    j_params["penalty_present"] = sp.penalty_present;
    j_params["penalty_freq"] = sp.penalty_freq;
    j_params["repeat_last_n"] = sp.penalty_last_n;
    j_params["repeat_penality"] = sp.penalty_repeat;
    j_params["mirostat"] = sp.mirostat;
    j_params["mirostat_tau"] = sp.mirostat_tau;
    j_params["mirostat_eta"] = sp.mirostat_eta;
    j_params["penalize_nl"] = sp.penalize_nl;

    return j_params;
}

static std::string now_timestr() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
    auto str = oss.str();
    return str;
}

struct PromptRunContext {
    int cur_test_nr;
    int total_tests;
    std::string test_id;
    json expected;
    int64_t seed;
    size_t prompt_token_cnt;

    PromptRunContext()
        : cur_test_nr(0), total_tests(0), seed(0), prompt_token_cnt(0) {}
};

json make_token_respose(PromptRunContext &prc, json tokens);

json make_token_respose(PromptRunContext &prc, json tokens) {
    if (prc.cur_test_nr <= 0) prc.cur_test_nr = 1;

    int passed_time = time(NULL) - benchmark_start_time;
    float time_per_test = ((float)passed_time) / (float)prc.cur_test_nr;
    float remaining =
        time_per_test * (float)(prc.total_tests - prc.cur_test_nr);
    remaining /= 60.0;

    float passed_time_mins = ((float)passed_time) / 60.0;

    std::string expected_s = "-";
    if (prc.expected.is_string())
        expected_s = prc.expected.template get<std::string>();

    std::string info_str = "[" + std::to_string(prc.cur_test_nr) + "/" +
                           std::to_string(prc.total_tests) +
                           "| id=" + prc.test_id +
                           " #p=" + std::to_string((int)prc.prompt_token_cnt) +
                           " #e=" + expected_s + "]: ";

    for (const auto &tok : tokens) {
        char buf[128];
        float prob = tok[1];
        std::string t = tok[0];
        snprintf(buf, 127, "[%s: %1.4f]", t.c_str(), prob);
        info_str += buf;
    }

    printf("[s/t=%5.2fs, eta=%5.1fm, t=%5.1fm] %s\n",
           time_per_test,
           remaining,
           passed_time_mins,
           info_str.c_str());
    fflush(stdout);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["tokens"] = tokens;
    single_response["expected"] = prc.expected;
    single_response["prompt_token_count"] = (int)prc.prompt_token_cnt;
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

json make_response(PromptRunContext &prc,
                   const std::string gen,
                   int gen_tok_cnt,
                   json prompt);

json make_response(PromptRunContext &prc,
                   const std::string gen,
                   int gen_tok_cnt,
                   json prompt) {
    if (prc.cur_test_nr <= 0) prc.cur_test_nr = 1;

    int passed_time = time(NULL) - benchmark_start_time;
    float time_per_test = ((float)passed_time) / (float)prc.cur_test_nr;
    float remaining =
        time_per_test * (float)(prc.total_tests - prc.cur_test_nr);
    remaining /= 60.0;

    float passed_time_mins = ((float)passed_time) / 60.0;

    std::string expected_s = "-";
    if (prc.expected.is_string())
        expected_s = prc.expected.template get<std::string>();

    std::string gen_prefix =
        "[" + std::to_string(prc.cur_test_nr) + "/" +
        std::to_string(prc.total_tests) + "| id=" + prc.test_id +
        ", seed=" + std::to_string(prc.seed) +
        ", #p=" + std::to_string((int)prc.prompt_token_cnt) +
        ", #g=" + std::to_string(gen_tok_cnt) + ", #e=" + expected_s + "]:";

    std::string print_gen = gen;
    std::replace(print_gen.begin(), print_gen.end(), '\r', ' ');
    std::replace(print_gen.begin(), print_gen.end(), '\n', '/');
    std::replace(print_gen.begin(), print_gen.end(), '\t', '/');

    printf("[s/t=%5.2fs, eta=%5.1fm, t=%5.1fm] %s %s\n",
           time_per_test,
           remaining,
           passed_time_mins,
           gen_prefix.c_str(),
           print_gen.c_str());
    fflush(stdout);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["seed"] = prc.seed;
    single_response["response"] = gen;
    single_response["expected"] = prc.expected;
    single_response["prompt"] = prompt;
    single_response["prompt_token_count"] = (int)prc.prompt_token_cnt;
    single_response["generated_token_count"] = gen_tok_cnt;
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

std::string trim_generated_chat_response(std::string gen);

std::string cleanup_unbalanced(const std::string &str,
                               char quot,
                               int &fixed_quote_len);
std::string cleanup_unbalanced(const std::string &str,
                               char quot,
                               int &fixed_quote_len) {
    bool found_open = false;
    size_t idx_last_quot = 0;

    for (size_t i = 0; i < str.size(); i++) {
        if (str[i] == quot) {
            idx_last_quot = i;
            found_open = !found_open;
        }
    }
    if (found_open) {
        std::string ini = str.substr(0, idx_last_quot);
        std::string o = str.substr(idx_last_quot + 1, std::string::npos);
        o = std::regex_replace(
            o, std::regex("^(.*[.?!,])[^.?!,]*$", std::regex::extended), "$1");
        trim_nl(o, " \r\n");
        fixed_quote_len = o.size();
        if (o.size() > 0) {
            auto last_char = o[o.size() - 1];
            if (last_char == ',') {
                o[o.size() - 1] = '.';
                if (quot == '"') {
                    o += "..";
                }
            } else if (quot == '"' && last_char != '?' && last_char != '!' &&
                       last_char != '.') {
                o += "...";
            }
        }
        if (o.size() == 0) {
            o = ini;
            trim_nl(o, " \r\n");
        } else {
            o = ini + quot + o + quot;
        }

        return o;
    }
    return str;
}

std::string trim_generated_chat_response(std::string gen) {
    // Strip extra newlines:
    gen = std::regex_replace(gen, std::regex("^:", std::regex::extended), "");
    int flen = 0;
    gen = cleanup_unbalanced(gen, '*', flen);
    gen = cleanup_unbalanced(gen, '"', flen);
    gen = std::regex_replace(
        gen, std::regex("\n\n\n*", std::regex::extended), "\n");
    // gen = std::regex_replace(
    //     gen, std::regex("\\.\\.\\.*", std::regex::extended), "");
    // gen = std::regex_replace(
    //     gen, std::regex("\\*\\*\\**", std::regex::extended), "");
    // Strip trailing cutted sentences:
    gen = std::regex_replace(
        gen,
        std::regex("^(.*[.!?*\")}`$])[^.!?*\")}`$]*$", std::regex::extended),
        "$1");
    trim_nl(gen, " \r\n");
    return gen;
}

struct TextReplacer {
    json prompt_runner_conf;
    json extra_replacements;

    TextReplacer() {}
    TextReplacer(json prconf) : prompt_runner_conf(prconf) {}

    void add_extra(const std::string &key, const std::string &repl) {
        extra_replacements[key] = repl;
    }

    void merge_replacements(json replacements) {
        if (replacements.find("replacements") != replacements.end()) {
            for (const auto &repl : replacements["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                extra_replacements[search] = replacement;
            }
        }
    }

    std::string apply_replacements(std::string prompt) {
        for (const auto &repl : extra_replacements.items()) {
            std::string search = repl.key();
            std::string replacement = repl.value();
            std::string new_t =
                std::regex_replace(prompt, std::regex(search), replacement);
            if (new_t != prompt) {
                prompt = apply_replacements(new_t);
            }
        }

        if (prompt_runner_conf.find("replacements") !=
            prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                std::string new_t =
                    std::regex_replace(prompt, std::regex(search), replacement);
                if (new_t != prompt) {
                    prompt = apply_replacements(new_t);
                }
            }
        }

        return prompt;
    }

    std::string apply_replacements(json prompt_test, std::string prompt) {
        for (const auto &repl : extra_replacements.items()) {
            std::string search = repl.key();
            std::string replacement = repl.value();
            std::string new_t =
                std::regex_replace(prompt, std::regex(search), replacement);
            if (new_t != prompt) {
                prompt = apply_replacements(prompt_test, new_t);
            }
        }

        if (prompt_runner_conf.find("replacements") !=
            prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                std::string new_t =
                    std::regex_replace(prompt, std::regex(search), replacement);
                if (new_t != prompt) {
                    prompt = apply_replacements(prompt_test, new_t);
                }
            }
        }

        if (prompt_test.find("replacements") != prompt_test.end()) {
            for (const auto &repl : prompt_test["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                std::string new_t =
                    std::regex_replace(prompt, std::regex(search), replacement);
                if (new_t != prompt) {
                    prompt = apply_replacements(prompt_test, new_t);
                }
            }
        }

        return prompt;
    }
};

struct Inference {
    struct Sequence {
        // [p0,p1) describing the position in kv cache
        int p0;
        int p1;
        // kv cache sequence ID
        int seq_id;
        // name of this sequence
        std::string name;
        // name of the previous sequence
        std::string prev_name;
        std::vector<llama_token> tokens;
        // for rewinding
        int recent_add_tokens;

        Sequence() : p0(0), p1(0), seq_id(0), recent_add_tokens(0) {}

        void commit() { recent_add_tokens = 0; }

        void add_token(llama_token tok, int new_p1 = -1) {
            if (new_p1 > 0) {
                if ((p1 + 1) != new_p1) {
                    printf(
                        "WARNING: Sequence gets appended unconnected token! %d "
                        "!= %d (%s)\n",
                        p1 + 1,
                        new_p1,
                        recent_string().c_str());
                }
                p1 = new_p1;
            } else {
                p1 += 1;
            }
            tokens.push_back(tok);
            recent_add_tokens += 1;
        }

        void rewind() {
            llama_kv_cache_seq_rm(*g_ctx, seq_id, p1 - recent_add_tokens, p1);
            for (int i = 0; i < recent_add_tokens; i++) {
                tokens.pop_back();
                p1 -= 1;
            }
            recent_add_tokens = 0;
        }

        std::string to_string() {
            std::string str;
            for (auto tok : tokens) {
                str += llama_token_to_piece(*g_ctx, tok);
            }
            return str;
        }

        std::string recent_string() {
            if (recent_add_tokens <= 0) {
                return "";
            }

            if (tokens.size() < (size_t)recent_add_tokens) {
                return "RECENT_TOKENS_BUGBUGBUG";
            }

            std::string str;
            for (size_t i = tokens.size() - recent_add_tokens;
                 i < tokens.size();
                 i++) {
                llama_token tok = tokens[i];
                str += llama_token_to_piece(*g_ctx, tok);
            }

            return str;
        }
    };

    int cur_seq_id;

    int n_batch;
    std::vector<Sequence> sequences;
    llama_sampling_context *ctx_sampling;

    Inference(llama_sampling_params &sparams, int nb)
        : cur_seq_id(0), n_batch(nb) {
        llama_kv_cache_clear(*g_ctx);
        ctx_sampling = llama_sampling_init(sparams);
    }

    void reset_seed(int seed) {
        // printf("RESET SEED %d\n", seed);
        llama_set_rng_seed(*g_ctx, seed);
    }

    void clear() {
        sequences.clear();
        llama_kv_cache_clear(*g_ctx);
        llama_sampling_reset(ctx_sampling);
    }

    ~Inference() {
        llama_kv_cache_clear(*g_ctx);
        llama_sampling_free(ctx_sampling);
    }

    int current_max_seq_token_count() {
        int max = 0;
        for (auto seq : sequences) {
            if (seq.p1 > max) {
                max = seq.p1;
            }
        }
        return max;
    }

    bool load_tokens(bool is_bos,
                     const std::string &text,
                     int seq_id,
                     int p0,
                     std::vector<llama_token> &out_tokens,
                     int &p1) {
        if (text.size() == 0) return true;

        const bool add_bos =
            is_bos && (llama_vocab_type(*g_model) == LLAMA_VOCAB_TYPE_SPM);
        std::vector<llama_token> tokens;
        tokens =
            ::llama_tokenize(*g_ctx, text.c_str(), add_bos, false, !is_bos);

        for (int i = 0; i < (int)tokens.size(); i += n_batch) {
            int n_eval = (int)tokens.size() - i;
            if (n_eval > n_batch) {
                n_eval = n_batch;
            }

            auto batch = llama_batch_init(tokens.size(), 0, 1);
            for (int j = 0; j < n_eval; j++) {
                int tok_idx = j + i;
                llama_batch_add(
                    batch, tokens[tok_idx], p0 + tok_idx, {seq_id}, false);
                out_tokens.push_back(tokens[tok_idx]);
                p1 = p0 + tok_idx + 1;
            }

            if (llama_decode(*g_ctx, batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return false;
            }

            llama_batch_free(batch);
        }

        return true;
    }

    bool add_start(const std::string &name, const std::string &text) {
        std::vector<llama_token> tokens;
        int p1 = 0;
        int seq_id = cur_seq_id++;

        bool ok = load_tokens(true, text, seq_id, 0, tokens, p1);
        if (ok) {
            Sequence seq;
            seq.p0 = 0;
            seq.p1 = p1;
            seq.seq_id = seq_id;
            seq.tokens = tokens;
            seq.name = name;
            seq.recent_add_tokens = tokens.size();
            sequences.push_back(seq);
        }

        return ok;
    }

    int get_sequence_idx(const std::string &name) {
        for (size_t i = 0; i < sequences.size(); i++) {
            if (sequences[i].name == name) {
                return (int)i;
            }
        }

        return -1;
    }

    void commit(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx >= 0) {
            sequences[sidx].commit();
        }
    }

    bool complete(const std::string &name,
                  const std::string &text,
                  int n_remain,
                  std::function<bool(int, const std::string &)> check_for_end) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }
        Sequence *seq = &sequences[sidx];
        //        printf("complete sequence: seq_id=%d p0=%d, p1=%d\n",
        //        seq->seq_id, seq->p0, seq->p1);

        auto tokens = ::llama_tokenize(*g_ctx, text.c_str(), false, true, true);
        auto batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            bool is_last = (i + 1 == tokens.size());
            // d// printf("ADDBATCH %d (is_last=%d) = %d\n", seq->p1, is_last,
            // tokens[i]);
            llama_batch_add(batch, tokens[i], seq->p1, {seq->seq_id}, is_last);
            seq->add_token(tokens[i]);
            llama_sampling_accept(ctx_sampling, *g_ctx, tokens[i], true);
            //            printf("accept %d\n", tokens[i]);
        }

        int sample_idx = tokens.size() - 1;

        if (llama_decode(*g_ctx, batch)) {
            fprintf(
                stderr, "%s [%s]: failed to eval\n", __func__, text.c_str());
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);

        printf(
            "COMPLETE SEQUENCE[%s: %s] seq_id=%d, p0=%d, p1=%d, size=%d, "
            "n_gen=%d\n",
            seq->name.c_str(),
            text.c_str(),
            seq->seq_id,
            seq->p0,
            seq->p1,
            (int)seq->tokens.size(),
            n_remain);

        int gen_count = 0;
        while (n_remain > 0) {
            llama_token tok = llama_sampling_sample(
                ctx_sampling, *g_ctx, nullptr, sample_idx);

            gen_count += 1;

            llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);
            //            printf("accept %d\n", tok);

            if (tok == llama_token_eos(*g_model)) {
                return true;
            }

            seq->add_token(tok);

            std::string seq_str = seq->recent_string();
            // d// printf("SEQ %d %s\n", seq->recent_add_tokens,
            // seq_str.c_str());
            if (check_for_end(gen_count, seq_str)) {
                n_remain = 0;
            } else {
                n_remain -= 1;
            }

            auto one_batch = llama_batch_init(tokens.size(), 0, 1);

            llama_batch_add(one_batch, tok, seq->p1 - 1, {seq->seq_id}, true);
            if (llama_decode(*g_ctx, one_batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                llama_batch_free(one_batch);
                return false;
            }
            sample_idx = 0;
            llama_batch_free(one_batch);
        }

        return true;
    }

    bool sample(
        const std::string &name,
        const std::string &text,
        int n_remain,
        std::function<llama_token(
            llama_token_data_array &, const std::string &, bool &)> sample) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }
        Sequence *seq = &sequences[sidx];

        auto tokens = ::llama_tokenize(*g_ctx, text.c_str(), false, true);
        auto batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            bool is_last = (i + 1 == tokens.size());
            llama_batch_add(batch, tokens[i], seq->p1, {seq->seq_id}, is_last);
            seq->add_token(tokens[i]);
            llama_sampling_accept(ctx_sampling, *g_ctx, tokens[i], true);
        }

        int sample_idx = tokens.size() - 1;

        if (llama_decode(*g_ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);

        // d// printf("COMPLETE SEQUENCE p0=%d, p1=%d, size=%d\n", seq->p0,
        // seq->p1, seq->tokens.size());

        int gen_count = 0;
        while (n_remain > 0) {
            gen_count += 1;

            float *logits = llama_get_logits_ith(*g_ctx, sample_idx);

            std::vector<llama_token_data> cur;
            const int n_vocab = llama_n_vocab(*g_model);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur.emplace_back(
                    llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {
                cur.data(), cur.size(), false};

            std::string seq_str = seq->recent_string();
            bool end = false;
            llama_token tok = sample(candidates_p, seq_str, end);
            if (tok == llama_token_eos(*g_model)) {
                return true;
            }
            if (end) {
                return true;
            }

            llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);

            seq->add_token(tok);

            auto one_batch = llama_batch_init(tokens.size(), 0, 1);

            llama_batch_add(one_batch, tok, seq->p1 - 1, {seq->seq_id}, true);
            if (llama_decode(*g_ctx, one_batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                llama_batch_free(one_batch);
                return false;
            }
            sample_idx = 0;
            llama_batch_free(one_batch);
        }

        return true;
    }

    std::string get_sequence_text_no_prev(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return "";
        }
        Sequence *seq = &sequences[sidx];

        return seq->to_string();
    }

    int get_sequence_token_count(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return 0;
        }
        Sequence *seq = &sequences[sidx];

        int prev_add = 0;
        std::string prev_name = sequences[sidx].prev_name;
        if (prev_name.size() > 0) {
            prev_add = get_sequence_token_count(prev_name);
        }

        return prev_add + (int)seq->tokens.size();
    }

    int get_sequence_token_count_no_prev(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return 0;
        }
        Sequence *seq = &sequences[sidx];

        return (int)seq->tokens.size();
    }

    std::string get_sequence_text(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return "";
        }
        Sequence *seq = &sequences[sidx];

        std::string add;
        std::string prev_name = sequences[sidx].prev_name;
        if (prev_name.size() > 0) {
            add = get_sequence_text(prev_name);
        }

        return add + seq->to_string();
    }

    std::string get_recently_added_tokens_str(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return "";
        }
        Sequence *seq = &sequences[sidx];

        return seq->recent_string();
    }

    bool rewind(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }
        Sequence *seq = &sequences[sidx];

        // llama_kv_cache_debug_print(*g_ctx, "bef_rewind");
        if (seq->recent_add_tokens > 0) {
            seq->rewind();
        }
        // llama_kv_cache_debug_print(*g_ctx, "aft_rewind");

        return true;
    }

    bool append(const std::string &name, const std::string &text) {
        int sidx = get_sequence_idx(name);
        Sequence *s = nullptr;
        if (sidx >= 0) {
            s = &sequences[sidx];
        } else {
            sequences.push_back(Sequence());
            s = &sequences[sequences.size() - 1];
            s->seq_id = cur_seq_id++;
        }

        // d// printf("SEQUENCE [%s] seq_id=%d\n", name.c_str(), s->seq_id);

        s->name = name;

        std::vector<llama_token> tokens;
        int new_p1 = 0;
        bool ok = load_tokens(false, text, s->seq_id, s->p1, tokens, new_p1);
        for (auto tok : tokens) {
            s->add_token(tok);
        }
        return ok;
    }

    void reset_sampler(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return;
        }
        llama_sampling_reset(ctx_sampling);

        std::string prev_name = sequences[sidx].prev_name;
        if (prev_name.size() > 0) {
            reset_sampler(prev_name);
        }

        for (auto tok : sequences[sidx].tokens) {
            llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);
            //            printf("accept[%s] %d\n", name.c_str(), tok);
        }
    }
};

struct StopSequences {
    json stop_sequences;

    void set_stop_sequences(json ss) { stop_sequences = ss; }

    bool trim_stop_sequence(const std::string &input, std::string &trimmed) {
        for (const auto &matched : stop_sequences) {
            size_t stop_pos = input.find(matched);
            if (stop_pos != std::string::npos) {
                trimmed = input.substr(0, stop_pos);
                return true;
            }
        }

        trimmed = input;

        return false;
    }
};

struct Character {
    TextReplacer replacer;
    std::string name;
    std::string prompt;
    std::string log_fmt;
    std::string next_fmt;
    std::string first_message;
    int n_gen_tokens;

    Character() : n_gen_tokens(0) {}

    void load(const TextReplacer &repl,
              json def,
              const std::string other_name,
              int n_predict_default) {
        replacer = repl;

        name = def.value("name", "Anon");

        replacer.merge_replacements(def);
        replacer.add_extra("\\{bot\\}", name);
        replacer.add_extra("\\{BOT\\}", name);
        replacer.add_extra("\\{char\\}", name);
        replacer.add_extra("\\{CHAR\\}", name);
        replacer.add_extra("\\{\\{char\\}\\}", name);
        replacer.add_extra("\\{\\{CHAR\\}\\}", name);
        replacer.add_extra("\\{user\\}", other_name);
        replacer.add_extra("\\{USER\\}", other_name);
        replacer.add_extra("\\{\\{user\\}\\}", other_name);
        replacer.add_extra("\\{\\{USER\\}\\}", other_name);
        replacer.add_extra("<BOT>", name);
        replacer.add_extra("<bot>", name);
        replacer.add_extra("<CHAR>", name);
        replacer.add_extra("<char>", name);
        replacer.add_extra("<USER>", other_name);
        replacer.add_extra("<user>", other_name);

        prompt = replacer.apply_replacements(def.value("prompt", ""));
        log_fmt = def.value("log_fmt", "<BOT>: <RESPONSE>\n");
        next_fmt = replacer.apply_replacements(def.value("next_fmt", "<BOT>:"));
        first_message = replacer.apply_replacements(def.value("first_mes", ""));
        n_gen_tokens = def.value("n_gen", n_predict_default);
    }

    std::string format_for_log(const std::string &response) {
        replacer.add_extra("<RESPONSE>", response);
        return replacer.apply_replacements(log_fmt);
    }

    std::string format_for_next() { return next_fmt; }
};

struct Conversation {
    struct ChatlogEntry {
        int prompt_token_count;
        int reroll_count;
        std::string text;
        std::string raw;

        ChatlogEntry() : prompt_token_count(0), reroll_count(0) {}
        ChatlogEntry(const std::string &txt) {
            prompt_token_count = 0;
            reroll_count = 0;
            text = txt;
            raw = txt;
        }
    };

    TextReplacer replacer;
    StopSequences stop_seq;

    json char_cfg;
    json user_cfg;

    Character c_user;
    Character c_char;

    int turns;

    std::vector<ChatlogEntry> chatlog;

    json test_replacements;

    Conversation(TextReplacer repl, StopSequences ss, json test_replacement)
        : replacer(repl), stop_seq(ss), test_replacements(test_replacement) {}

    bool load_config(json chat, int n_predict_default) {
        if (chat.contains("user")) {
            user_cfg = chat["user"];
        } else {
            fprintf(stderr, "BAD CHAT, NO \"user\" KEY!\n");
            return false;
        }

        if (chat.contains("char")) {
            char_cfg = chat["char"];
        } else {
            fprintf(stderr, "BAD CHAT, NO \"user\" KEY!\n");
            return false;
        }

        replacer.merge_replacements(test_replacements);

        c_char.load(replacer,
                    char_cfg,
                    user_cfg.value("name", "Ayumi"),
                    n_predict_default);
        c_user.load(replacer,
                    user_cfg,
                    char_cfg.value("name", "Anon"),
                    n_predict_default);

        turns = chat.value("turns", 100);

        if (c_char.first_message != "") {
            std::string entry = c_char.format_for_log(c_char.first_message);
            chatlog.push_back(ChatlogEntry(entry));
        }

        return true;
    }

    int chat_turns() { return turns; }

    std::string next_completion_fmt(bool is_user) {
        return is_user ? c_user.format_for_next() : c_char.format_for_next();
    }

    int next_completion_tok_count(bool is_user) {
        return is_user ? c_user.n_gen_tokens : c_char.n_gen_tokens;
    }

    std::string append_raw_chat_response(bool is_user,
                                         const std::string &resp,
                                         const std::string &raw,
                                         int reroll_count,
                                         int prompt_token_count) {
        std::string cleaned;
        stop_seq.trim_stop_sequence(resp, cleaned);
        cleaned = trim_generated_chat_response(cleaned);

        trim_nl(cleaned, " \r\n");

        if (cleaned.size() == 0) {
            return "";
        }

        Character *c = is_user ? &c_user : &c_char;
        std::string entry = c->format_for_log(cleaned);

        ChatlogEntry ce;
        ce.prompt_token_count = prompt_token_count;
        ce.text = entry;
        ce.raw = raw;
        ce.reroll_count = reroll_count;
        chatlog.push_back(ce);

        return entry;
    }

    std::string concatl(const std::vector<ChatlogEntry> &l, bool indent) {
        std::string logstr;
        for (auto logentry : l) {
            if (indent) {
                std::string le = logentry.text;
                std::replace(le.begin(), le.end(), '\n', ' ');
                logstr += "    " + le + "\n\n";
            } else {
                logstr += logentry.text;
            }
        }
        return logstr;
    }

    std::string user_prompt() { return c_user.prompt; }
    std::string char_prompt() { return c_char.prompt; }

    std::string chatlog_text() { return concatl(chatlog, false); }

    std::string chatlog_text_ext() { return concatl(chatlog, true); }

    json raw_chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            log.push_back(l.raw);
        }
        return log;
    }

    json text_chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            log.push_back(l.text);
        }
        return log;
    }

    json chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            json entry;
            entry["prompt_token_count"] = l.prompt_token_count;
            entry["reroll_count"] = l.reroll_count;
            entry["text"] = l.text;
            entry["raw"] = l.raw;
            log.push_back(entry);
        }
        return log;
    }

    void log_chatlog_to_file(int seed,
                             const std::string &model_file,
                             json prompt_test,
                             llama_sampling_params &sp) {
        std::string test_id = prompt_test.value("id", "unknown_test_id");
        json sinfo = sparams_to_json(sp);

        std::string out_file_name = "chatlog_" + model_file + "_" + test_id +
                                    "_" + std::to_string(time(NULL)) + ".md";
        std::ofstream outf(out_file_name);
        outf << "# Chatlog for Test ID " << test_id << ", seed " << seed
             << "\n\n";
        outf << "Model Filename: " << model_file << "\n\n";
        outf << "## Sampling Parameters\n\n```json\n";
        outf << sinfo.dump(2, ' ', false, json::error_handler_t::replace);
        outf << "\n```\n\n";
        outf << "## Character Prompt '" << c_char.name << "'\n\n";
        outf << "```\n";
        outf << c_char.prompt;
        outf << "\n```\n\n";
        outf << "## User Prompt '" << c_user.name << "'\n\n";
        outf << "```\n";
        outf << c_user.prompt;
        outf << "\n```\n\n";
        outf << "## Chatlog\n\n";
        outf << chatlog_text_ext();
        outf << "\n";
        outf.close();
        printf("WROTE FILE %s\n", out_file_name.c_str());
        fflush(stdout);

        json chatlog;
        chatlog["model_file"] = model_file;
        chatlog["sampling"] = sinfo;
        chatlog["chatlog"] = chatlog_as_json();
        chatlog["timestamp"] = std::to_string(time(NULL));
        chatlog["time"] = now_timestr();
        std::string jout_file_name = "chatlog_" + model_file + "_" + test_id +
                                     "_" + std::to_string(time(NULL)) + ".json";
        std::ofstream joutf(jout_file_name);
        joutf << chatlog.dump(2, ' ', false, json::error_handler_t::replace);
        joutf.close();
        printf("WROTE FILE %s\n", jout_file_name.c_str());
        fflush(stdout);
    }
};

void record_token_info(json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling,
                       int sample_idx);

void record_token_info(json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling,
                       int sample_idx) {
    const int n_vocab = llama_n_vocab(*g_model);

    float *logits = llama_get_logits_ith(*g_ctx, sample_idx);

    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = {cur.data(), cur.size(), false};

    llama_sample_grammar(*g_ctx, &candidates_p, ctx_sampling->grammar);

    // Explicitly refuse the " " token.
    for (size_t i = 0; i < candidates_p.size; ++i) {
        const llama_token id = candidates_p.data[i].id;
        const std::string piece = llama_token_to_piece(*g_ctx, id);
        if (piece == " ") {
            candidates_p.data[i].logit = -INFINITY;
        }
    }

    llama_sample_softmax(nullptr, &candidates_p);

    json tokens;

    for (size_t i = 0; i < candidates_p.size; ++i) {
        if (candidates_p.data[i].p > 0.00001) {
            std::string tok = tokens_to_output_formatted_string(
                *g_ctx, candidates_p.data[i].id);
            json j_tok;
            j_tok[0] = tok;
            j_tok[1] = candidates_p.data[i].p;
            tokens.push_back(j_tok);
        }
    }

    if (tokens.size() > 1) {
        j_tok_resps.push_back(make_token_respose(prc, tokens));
    }
}

llama_token generate_sample_seeded(json &j_resps,
                                   std::vector<int64_t> &sample_seeds,
                                   PromptRunContext &prc,
                                   struct llama_sampling_context *ctx_sampling);

llama_token generate_sample_seeded(
    json j_resps,
    std::vector<int64_t> &sample_seeds,
    PromptRunContext &prc,
    struct llama_sampling_context *ctx_sampling) {
    llama_token id = 0;
    for (auto seed : sample_seeds) {
        llama_set_rng_seed(*g_ctx, seed);
        prc.seed = seed;

        id = llama_sampling_sample(ctx_sampling, *g_ctx, nullptr);

        if (sample_seeds.size() > 0 && id != llama_token_eos(*g_model)) {
            std::string gen = tokens_to_output_formatted_string(*g_ctx, id);
            j_resps.push_back(make_response(prc, gen, 1, json()));
        }
    }

    return id;
}

bool chatlog_has_repetitions(const std::vector<std::string> &chatlog,
                             std::string &piece,
                             std::string &prevlog);

bool chatlog_has_repetitions(const std::vector<std::string> &chatlog,
                             std::string &piece,
                             std::string &prevlog) {
    if (chatlog.size() < 2) {
        return false;
    }

    std::string last = chatlog[chatlog.size() - 1];
    if (last.size() < 30) {
        return false;
    }

    // if 30 characters appear verbatim in a previous log entry:
    int part_repetition_size = 30;
    if (part_repetition_size > (int)last.size()) {
        part_repetition_size = (int)last.size();
    }
    if (((int)last.size()) > part_repetition_size) {
        // 75% repetition 1:1
        part_repetition_size = ((int)last.size()) * 0.75;
    }

    for (int i = 2; i < (int)chatlog.size(); i++) {
        std::string preventry = chatlog[chatlog.size() - i];
        int diff = ((int)preventry.size()) - (int)last.size();
        if (diff < 0) {
            diff = -1 * diff;
        }
        if (diff > 30) {
            // messages too different in size!
            continue;
        }

        int extra = ((int)last.size()) - part_repetition_size;
        if (extra < 0) {
            extra = 0;
        }
        for (int offs = 0; offs < extra; offs++) {
            std::string subpart = last.substr(offs, part_repetition_size);
            if (preventry.find(subpart) != std::string::npos) {
                piece = subpart;
                prevlog = preventry;
                return true;
            }
        }
    }

    return false;
}

// "chat": {
//   "user": {
//       "prompt": "<PROMPT2><CHATLOG>Loki: ",
//       "n_gen": 70,
//       "log_fmt": "Loki: <RESPONSE>\n"
//   },
//   "char": {
//       "prompt": "<PROMPT><CHATLOG>Aria: ",
//       "n_gen": 70,
//       "log_fmt": "Aria: <RESPONSE>\n"
//   },
//   "char_log_init": "Arias character and scenario
//   described here.", "turns": 50
// },
bool chatlog_generator(PromptRunContext &prun_ctx,
                       gpt_params &params,
                       Inference &infer,
                       TextReplacer &replacer,
                       json prompt_runner_conf,
                       json prompt_test,
                       json &j_resps);

bool chatlog_generator(PromptRunContext &prun_ctx,
                       gpt_params &params,
                       Inference &infer,
                       TextReplacer &replacer,
                       json prompt_runner_conf,
                       json prompt_test,
                       json &j_resps) {
    StopSequences stop_seq;
    if (prompt_runner_conf.find("stop_sequences") != prompt_runner_conf.end()) {
        stop_seq.set_stop_sequences(prompt_runner_conf["stop_sequences"]);
    }

    Conversation conversation(replacer, stop_seq, prompt_test);
    conversation.load_config(prompt_test["chat"], params.n_predict);

    if (!infer.add_start("char", conversation.char_prompt())) {
        fprintf(stderr, "Couldn't add_start char prompt\n");
        fflush(stderr);
        return false;
    }
    if (!infer.add_start("user", conversation.user_prompt())) {
        fprintf(stderr, "Couldn't add_start user prompt\n");
        fflush(stderr);
        return false;
    }

    if (!infer.append("char", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }
    infer.commit("char");

    if (!infer.append("user", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }
    infer.commit("user");

    bool is_user = true;
    std::string end_reason;

    int prompt_max_len = 3900;

    int rerolled_broken_quotes = 0;
    int rerolled_empty_replies = 0;
    int reroll = 0;
    for (int i = 0; i < conversation.chat_turns(); i++) {
        bool user_max_tokens =
            infer.get_sequence_token_count("user") > prompt_max_len;
        bool char_max_tokens =
            infer.get_sequence_token_count("char") > prompt_max_len;
        if (user_max_tokens && char_max_tokens) {
            end_reason = "context limit reached (" +
                         std::to_string(prompt_max_len) + ")";
            break;
        }

        if (reroll > 10) {
            end_reason = "reroll limit reached (10)";
            break;
        }

        std::string sequence_name = is_user ? "user" : "char";

        printf("SEQ[log %d]=%s\n",
               infer.current_max_seq_token_count(),
               infer.get_sequence_text(sequence_name).c_str());
        fflush(stdout);

        infer.reset_seed(prun_ctx.seed + i + reroll);
        infer.reset_sampler(sequence_name);

        std::string completion_start =
            conversation.next_completion_fmt(is_user);
        int completion_token_cnt =
            conversation.next_completion_tok_count(is_user);

        std::string user_first_msg = "";
        if (conversation.c_user.first_message != "" && i < 2 && is_user) {
            user_first_msg = conversation.c_user.first_message;
            completion_start = completion_start + " " + user_first_msg;
        }
        printf("COMPLSTART[%s]\n", completion_start.c_str());

        if (!infer.complete(sequence_name,
                            completion_start,
                            completion_token_cnt,
                            [&](int, const std::string &comp) {
                                std::string tr;
                                return stop_seq.trim_stop_sequence(
                                    comp.substr(completion_start.size()), tr);
                            })) {
            end_reason = "inference error";
            fprintf(stderr, "Inference error!\n");
            fflush(stderr);
            return false;
            break;
        }

        int prompt_token_count = infer.get_sequence_token_count(sequence_name);
        std::string completion =
            infer.get_recently_added_tokens_str(sequence_name);
        std::string raw = completion;
        completion = completion.substr(completion_start.size());
        if (is_user && user_first_msg != "") {
            completion = user_first_msg + completion;
        }
        infer.rewind(sequence_name);
        printf("COMPLEND[%s]\n", completion.c_str());

        int fixed_quote_len = -1;
        cleanup_unbalanced(completion, '"', fixed_quote_len);
        if (fixed_quote_len > -1 && fixed_quote_len < 5) {
            printf("REROLL\n");
            rerolled_broken_quotes++;
            reroll++;
            i--;
            continue;
        }

        std::string log_entry = conversation.append_raw_chat_response(
            is_user, completion, raw, reroll, prompt_token_count);

        if (log_entry == "") {
            printf("REROLL\n");
            rerolled_empty_replies++;
            reroll++;
            i--;
            continue;
        }

        std::string print_gen = log_entry;
        std::replace(print_gen.begin(), print_gen.end(), '\r', ' ');
        std::replace(print_gen.begin(), print_gen.end(), '\n', '/');
        std::replace(print_gen.begin(), print_gen.end(), '\t', '/');

        float chat_fraction = ((float)i) / ((float)conversation.chat_turns());
        int passed_time = time(NULL) - benchmark_start_time;
        float passed_tests_f =
            (((float)(prun_ctx.cur_test_nr - 1)) + chat_fraction);
        float time_per_test =
            passed_tests_f > 0.01 ? ((float)passed_time) / passed_tests_f : 0.0;
        float remaining =
            time_per_test * (((float)prun_ctx.total_tests) - passed_tests_f);
        remaining /= 60.0;

        float passed_time_mins = ((float)passed_time) / 60.0;

        printf(
            "[test_id=%s, eta=%5.1fm, t=%5.1fm, seed=%ld, test_nr=%d, "
            "total_tests=%d, cur_turn=%d, turns=%d, plen=%d, pmax=%d]: %s\n",
            prun_ctx.test_id.c_str(),
            remaining,
            passed_time_mins,
            prun_ctx.seed,
            prun_ctx.cur_test_nr,
            prun_ctx.total_tests,
            i,
            conversation.chat_turns(),
            infer.current_max_seq_token_count(),
            prompt_max_len,
            print_gen.c_str());
        fflush(stdout);

        if (completion.size() == 0) {
            end_reason = "empty response";
            break;
        }

        if (!infer.append("user", log_entry)) {
            fprintf(stderr, "Couldn't append chatlog\n");
            fflush(stderr);
            return false;
        }
        infer.commit("user");

        if (!infer.append("char", log_entry)) {
            fprintf(stderr, "Couldn't append chatlog\n");
            fflush(stderr);
            return false;
        }
        infer.commit("char");

        is_user = !is_user;
    }

    llama_sampling_params &sparams = params.sparams;

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    conversation.log_chatlog_to_file(
        prun_ctx.seed, model_file, prompt_test, sparams);

    json prompt_collection;
    prompt_collection["char"] = infer.get_sequence_text("user");
    prompt_collection["user"] = infer.get_sequence_text("char");
    prompt_collection["chatlog"] = conversation.chatlog_as_json();
    prompt_collection["raw_chatlog"] = conversation.raw_chatlog_as_json();
    prompt_collection["text_chatlog"] = conversation.text_chatlog_as_json();
    prompt_collection["max_token_count"] = infer.current_max_seq_token_count();
    prompt_collection["rerolled_broken_quotes"] = rerolled_broken_quotes;
    prompt_collection["rerolled_empty_replies"] = rerolled_empty_replies;
    prompt_collection["end_reason"] = end_reason;

    j_resps.push_back(make_response(prun_ctx,
                                    conversation.chatlog_text(),
                                    infer.current_max_seq_token_count(),
                                    prompt_collection));

    return true;
}

int main(int argc, char **argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    llama_sampling_params &sparams = params.sparams;

    if (!sparams.grammar.empty()) {
        grammar_parser::parse_state parsed_grammar;
        parsed_grammar = grammar_parser::parse(sparams.grammar.c_str());

        fprintf(stderr, "%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    const json prompt_runner_conf = json::parse(params.prompt);

    if (prompt_runner_conf.find("prompt_tests") == prompt_runner_conf.end()) {
        fprintf(stderr, "**********\n");
        fprintf(stderr,
                "ERROR: No prompt_tests in prompt_runner_config.json!\n");
        fprintf(stderr, "**********\n");

        return 1;
    }

    LOG_TEE(
        "%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n",
            __func__,
            LLAMA_COMPILER,
            LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model *model;
    llama_context *ctx;
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
        LOG_TEE(
            "%s: warning: model was trained on only %d context tokens (%d "
            "specified)\n",
            __func__,
            n_ctx_train,
            n_ctx);
    }

    if (params.n_ctx < 8) {
        fprintf(stderr,
                "%s: warning: minimum context size is 8, using minimum size.\n",
                __func__);
        params.n_ctx = 8;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr,
                "system_info: n_threads = %d / %d | %s\n",
                params.n_threads,
                std::thread::hardware_concurrency(),
                llama_print_system_info());
    }

    // Add BOS if SPM tokenizer
    const bool add_bos = llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM;

    // tokenize the prompt
    std::vector<llama_token> embd_inp;
    json j_resps;
    json j_tok_resps;

    bool first = true;

    bool record_next_token_info =
        prompt_runner_conf.value("record_next_token_info", false);

    std::vector<int64_t> sample_seeds;
    if (prompt_runner_conf.find("sample_seeds") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading sample_seeds\n");
        for (const auto &seed_value : prompt_runner_conf["sample_seeds"]) {
            int64_t seed = seed_value;
            sample_seeds.push_back(seed);
        }
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
        fprintf(
            stderr,
            "No \"prompt_tests\" defined in the prompt_runner_config.json!\n");
        return 1;
    }

    benchmark_start_time = time(NULL);

    fprintf(stderr, "PROMPT-RUNNER-START\n");
    fflush(stderr);

    TextReplacer replacer(prompt_runner_conf);

    PromptRunContext prun_ctx;
    prun_ctx.total_tests = seeds.size() * test_count;

    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        json expected;
        if (prompt_test.find("expected") != prompt_test.end()) {
            expected = prompt_test["expected"];
        }
        prun_ctx.expected = expected;

        prun_ctx.test_id = prompt_test.value("id", "unknown_test_id");

        std::string prompt =
            replacer.apply_replacements(prompt_test, "<PROMPT>");
        rtrim_nl(prompt);

        if (params.verbose_prompt) {
            printf(
                "PROMPT------------------------\n%s\n--------------------------"
                "---\n",
                prompt.c_str());
        }

        fflush(stdout);

        for (const auto &seed_value : seeds) {
            prun_ctx.seed = seed_value;
            llama_set_rng_seed(ctx, prun_ctx.seed);

            prun_ctx.cur_test_nr += 1;

            if (params.prompt.empty()) {
                fprintf(stderr, "No prompt given!");
                return 1;
            }

            embd_inp = ::llama_tokenize(ctx, prompt.c_str(), add_bos, true);

            const int n_ctx = llama_n_ctx(ctx);
            prun_ctx.prompt_token_cnt = embd_inp.size();

            if (((int)embd_inp.size() + (int)params.n_predict) > n_ctx - 4) {
                fprintf(stderr,
                        "%s: error: prompt is too long (%d tokens, %d "
                        "predict, max %d)\n",
                        __func__,
                        (int)embd_inp.size(),
                        (int)params.n_predict,
                        n_ctx - 4);
                return 1;
            }

            if (first) {
                fprintf(stderr,
                        "sampling: \n%s\n",
                        llama_sampling_print(sparams).c_str());
            }

            if (prompt_test.find("query") != prompt_test.end()) {
                // "query": {
                //   "sampling": { "tfs-z": 0.95, "temp": 0.9 } },
                //   "replacements": [
                //      ["<U>", "<USER>:"],
                //      ["<C>", "<USER>:"],
                //   ],
                //   "messages": [
                //       {"msg":"<C> *<CHAR> sits in a library*"},
                //       {"msg":"<U> Hey <CHAR> *waves* I heard you had
                //       birthday, how old did you get?"},
                //       {"msg":"<C> Hi <USER>, yes I got",
                //        "query_id": "age"
                //        "msg_postfix": "years old.",
                //        "complete": { "n_gen": 10, "bnf": "\" \"?
                //        [1-9][0-9]*" },
                //       },
                //       {"msg":"<U> Amazing! You got new clothes I see,
                //       what are you wearing?"},
                //       {"msg":"<C> Right now I am wearing",
                //        "query_id": "clothes"
                //        "msg_postfix": ".",
                //        "complete": { "n_gen": 10, "top-k": 10, "dfs":
                //        true }
                //       },
                //   ],
                // }
                //
                // "age" would result in a multiplied probability of the
                // resulting answer. "clothes" would be a list of answers,
                // each with their multiplied probabilty.

            } else if (prompt_test.find("chat") != prompt_test.end()) {
                Inference infer(sparams, params.n_batch);

                if (!chatlog_generator(prun_ctx,
                                       params,
                                       infer,
                                       replacer,
                                       prompt_runner_conf,
                                       prompt_test,
                                       j_resps)) {
                    return 1;
                }

            } else {
            }

            first = false;
        }
    }

    llama_print_timings(ctx);

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    printf("model: %s\n", model_file.c_str());

    json j_params = sparams_to_json(sparams);
    j_params["rope_freq_base"] = params.rope_freq_base;
    j_params["rope_freq_scale"] = params.rope_freq_scale;

    json results;
    results["params"] = j_params;
    results["model_file"] = model_file;
    results["prompt"] = std::string(params.prompt);
    results["config"] = prompt_runner_conf;
    results["results"] = j_resps;
    if (record_next_token_info) {
        results["tokens"] = j_tok_resps;
    }

    std::string out_file_name =
        "result_" + std::to_string(time(NULL)) + "_" + model_file + ".json";
    std::ofstream outf(out_file_name);
    outf << results.dump(2, ' ', false, json::error_handler_t::replace);
    outf.close();

    printf("[PROMPT_RUNNER_OUTPUT_FILE: %s]\n", out_file_name.c_str());
    fflush(stdout);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
