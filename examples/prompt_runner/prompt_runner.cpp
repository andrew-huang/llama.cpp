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

#include "build-info.h"
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
    float temp;
    int64_t seed;
    size_t prompt_token_cnt;

    PromptRunContext()
        : cur_test_nr(0),
          total_tests(0),
          temp(1.0),
          seed(0),
          prompt_token_cnt(0) {}
};

json make_token_respose(std::vector<std::string> &responses,
                        PromptRunContext &prc,
                        json tokens);

json make_token_respose(std::vector<std::string> &responses,
                        PromptRunContext &prc,
                        json tokens) {
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

    responses.push_back(prc.test_id + "=" + info_str);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["tokens"] = tokens;
    single_response["expected"] = prc.expected;
    single_response["prompt_token_count"] = (int)prc.prompt_token_cnt;
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

json make_response(std::vector<std::string> &responses,
                   PromptRunContext &prc,
                   const std::string gen,
                   int gen_tok_cnt,
                   json prompt);

json make_response(std::vector<std::string> &responses,
                   PromptRunContext &prc,
                   const std::string gen,
                   int gen_tok_cnt,
                   json prompt) {
    std::ostringstream oss;
    oss << std::setprecision(1) << prc.temp;
    std::string temp_str = oss.str();

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
        ", temp=" + temp_str + ", seed=" + std::to_string(prc.seed) +
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

    responses.push_back(gen_prefix + gen);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["seed"] = prc.seed;
    single_response["temp"] = temp_str;
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

std::string cleanup_unbalanced(const std::string &str, char quot) {
    bool found_open = false;
    size_t idx_last_quot = 0;
    for (size_t i = 0; i < str.size(); i++) {
        if (str[i] == quot) {
            idx_last_quot = i;
            found_open = !found_open;
        }
    }
    if (found_open) {
        std::string tmp = str.substr(0, idx_last_quot);
        trim_nl(tmp, " \r\n");
        if (tmp.size() == 0) {
            // removing the unbalanced part would make the result empty!
            // so we append the unbalanced quot instead.
            tmp = str + quot;
        }

        return tmp;
    }
    return str;
}

std::string trim_generated_chat_response(std::string gen) {
    // Strip extra newlines:
    gen = std::regex_replace(gen, std::regex("^:", std::regex::extended), "");
    gen = std::regex_replace(
        gen, std::regex("\n\n\n*", std::regex::extended), "\n");
    // gen = std::regex_replace(
    //     gen, std::regex("\\.\\.\\.*", std::regex::extended), "");
    // gen = std::regex_replace(
    //     gen, std::regex("\\*\\*\\**", std::regex::extended), "");
    // Strip trailing cutted sentences:
    gen = std::regex_replace(
        gen,
        std::regex("(.*[.!?*\")}`$])[^.!?*\")}`$]*", std::regex::extended),
        "$1");
    gen = cleanup_unbalanced(gen, '*');
    gen = cleanup_unbalanced(gen, '"');
    trim_nl(gen, " \r\n");
    return gen;
}

struct TextReplacer {
    json prompt_runner_conf;
    json extra_replacements;

    TextReplacer(json prconf) : prompt_runner_conf(prconf) {}

    void add_extra(const std::string &key, const std::string &repl) {
        extra_replacements[key] = repl;
    }

    std::string apply_replacements(json prompt_test, std::string prompt) {
        for (const auto &repl : extra_replacements.items()) {
            std::string search = repl.key();
            std::string replacement = repl.value();
            prompt =
                std::regex_replace(prompt, std::regex(search), replacement);
        }

        if (prompt_runner_conf.find("replacements") !=
            prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                prompt =
                    std::regex_replace(prompt, std::regex(search), replacement);
            }
        }

        std::string repl_info = "";

        if (prompt_test.find("replacements") != prompt_test.end()) {
            for (const auto &repl : prompt_test["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                if (replacement.size() < 250) {
                    repl_info += search + " := " + replacement + "\n";
                }
                prompt =
                    std::regex_replace(prompt, std::regex(search), replacement);
            }
        }

        if (repl_info.size() > 0) {
            printf("------------------------\n%s", repl_info.c_str());
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
        // sequence ID before rebase
        int rebase_seq_id;
        // name of this sequence
        std::string name;
        // name of the previous sequence
        std::string prev_name;
        std::vector<llama_token> tokens;
        // for rewinding
        int recent_add_tokens;

        Sequence()
            : p0(0),
              p1(0),
              seq_id(0),
              rebase_seq_id(-1),
              recent_add_tokens(0) {}

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
        std::string t = add_bos ? std::string(" ") + text : text;
        auto tokens = ::llama_tokenize(*g_ctx, t.c_str(), add_bos, true);

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
            printf("SEQUENCEb[%s] seq_id=%d\n", name.c_str(), seq.seq_id);
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

    bool rebase(const std::string &name, const std::string &new_prev) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }

        int dest_idx = get_sequence_idx(new_prev);
        if (dest_idx < 0) {
            return false;
        }

        Sequence *src = &sequences[sidx];
        Sequence *dest = &sequences[dest_idx];
        printf("REBASE [%s]seq=%d [%d-%d) onto [%s]seq=%d [%d-%d)\n",
               src->name.c_str(),
               src->seq_id,
               src->p0,
               src->p1,
               dest->name.c_str(),
               dest->seq_id,
               dest->p0,
               dest->p1);

        llama_kv_cache_debug_print(*g_ctx, "bef_reb");
        if (src->rebase_seq_id < 0) {
            src->rebase_seq_id = src->seq_id;
        } else {
            //            llama_kv_cache_debug_print(*g_ctx, "pre_copy");
            llama_kv_cache_seq_cp(
                *g_ctx, src->seq_id, src->rebase_seq_id, src->p0, src->p1);
            llama_kv_cache_seq_rm(*g_ctx, src->seq_id, src->p0, src->p1);
            src->seq_id = src->rebase_seq_id;
        }

        //        llama_kv_cache_debug_print(*g_ctx, "before");
        int offs = dest->p1 - src->p0;
        llama_kv_cache_seq_shift(*g_ctx, src->seq_id, src->p0, src->p1, offs);
        //        llama_kv_cache_debug_print(*g_ctx, "aft_shift");
        src->p0 += offs;
        src->p1 += offs;
        src->prev_name = new_prev;

        llama_kv_cache_seq_cp(
            *g_ctx, src->seq_id, dest->seq_id, src->p0, src->p1);
        //        llama_kv_cache_debug_print(*g_ctx, "aft_cp");
        llama_kv_cache_seq_rm(*g_ctx, src->seq_id, src->p0, src->p1);
        //        llama_kv_cache_debug_print(*g_ctx, "after");
        llama_kv_cache_debug_print(*g_ctx, "aft_reb");

        src->seq_id = dest->seq_id;

        return true;
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
        printf("complete sequence: seq_id=%d p0=%d, p1=%d\n",
               seq->seq_id,
               seq->p0,
               seq->p1);

        auto tokens = ::llama_tokenize(*g_ctx, text.c_str(), false, true);
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

        llama_kv_cache_debug_print(*g_ctx, "bef_rewind");
        if (seq->recent_add_tokens > 0) {
            seq->rewind();
        }
        llama_kv_cache_debug_print(*g_ctx, "aft_rewind");

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
        printf("APPENDING[%s]\n", text.c_str());

        // d// printf("SEQUENCE [%s] seq_id=%d\n", name.c_str(), s->seq_id);

        s->name = name;

        std::vector<llama_token> tokens;
        int new_p1 = 0;
        bool ok = load_tokens(false, text, s->seq_id, s->p1, tokens, new_p1);
        for (auto tok : tokens) {
            s->add_token(tok);
        }
        printf("append to seq_id=%d new_p1=%d, seq->p1=%d [%s]\n",
               s->seq_id,
               new_p1,
               s->p1,
               s->recent_string().c_str());
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

struct Conversation {
    TextReplacer replacer;
    StopSequences stop_seq;
    std::string char_log_fmt;
    std::string user_log_fmt;

    std::string char_prompt;
    std::string user_prompt;

    std::string char_next_fmt;
    std::string user_next_fmt;
    int user_next_tok;
    int char_next_tok;

    int turns;

    json char_cfg;
    json user_cfg;

    std::vector<std::string> chatlog;

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

        user_prompt = replacer.apply_replacements(test_replacements,
                                                  user_cfg.value("prompt", ""));
        char_prompt = replacer.apply_replacements(test_replacements,
                                                  char_cfg.value("prompt", ""));
        user_log_fmt = user_cfg.value("log_fmt", "<USER>: <RESPONSE>\n");
        char_log_fmt = char_cfg.value("log_fmt", "<CHAR>: <RESPONSE>\n");
        user_next_fmt = replacer.apply_replacements(
            test_replacements, user_cfg.value("next_fmt", "<USER>:"));
        char_next_fmt = replacer.apply_replacements(
            test_replacements, char_cfg.value("next_fmt", "<CHAR>:"));
        user_next_tok = user_cfg.value("n_gen", n_predict_default);
        char_next_tok = user_cfg.value("n_gen", n_predict_default);

        turns = chat.value("turns", 100);

        std::string char_log_init = chat.value("char_log_init", "");
        if (char_log_init.size() > 0) {
            replacer.add_extra("<RESPONSE>", char_log_init);
            std::string entry =
                replacer.apply_replacements(test_replacements, char_log_fmt);
            chatlog.push_back(entry);
        }

        return true;
    }

    int chat_turns() { return turns; }

    std::string next_completion_fmt(bool is_user) {
        return is_user ? user_next_fmt : char_next_fmt;
    }

    int next_completion_tok_count(bool is_user) {
        return is_user ? user_next_tok : char_next_tok;
    }

    std::string append_raw_chat_response(bool is_user,
                                         const std::string &resp) {
        std::string cleaned;
        stop_seq.trim_stop_sequence(resp, cleaned);
        cleaned = trim_generated_chat_response(cleaned);

        trim_nl(cleaned, " \r\n");
        replacer.add_extra("<RESPONSE>", cleaned);
        std::string entry = replacer.apply_replacements(
            test_replacements, is_user ? user_log_fmt : char_log_fmt);
        chatlog.push_back(entry);

        // d// printf("APPENDLOG:[%s]\n", chatlog.back().c_str());

        return entry;
    }

    std::string chatlog_text() { return concatl(chatlog); }

    std::string chatlog_text_ext() { return concatl(chatlog, true); }

    json chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            log.push_back(l);
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
        outf << "## Character Prompt '"
             << replacer.apply_replacements(test_replacements, "<CHAR>")
             << "'\n\n";
        outf << "```\n";
        outf << char_prompt;
        outf << "\n```\n\n";
        outf << "## User Prompt '"
             << replacer.apply_replacements(test_replacements, "<USER>")
             << "'\n\n";
        outf << "```\n";
        outf << user_prompt;
        outf << "\n```\n\n";
        outf << "## Chatlog\n\n";
        outf << chatlog_text_ext();
        outf << "\n";
        outf.close();

        printf("wrote file %s\n", out_file_name.c_str());
    }
};

void record_token_info(std::vector<std::string> &responses,
                       json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling,
                       int sample_idx);

void record_token_info(std::vector<std::string> &responses,
                       json &j_tok_resps,
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
        j_tok_resps.push_back(make_token_respose(responses, prc, tokens));
    }
}

llama_token generate_sample_seeded(std::vector<std::string> &responses,
                                   json j_resps,
                                   std::vector<int64_t> &sample_seeds,
                                   PromptRunContext &prc,
                                   struct llama_sampling_context *ctx_sampling);

llama_token generate_sample_seeded(
    std::vector<std::string> &responses,
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
            j_resps.push_back(make_response(responses, prc, gen, 1, json()));
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
                       json j_resps,
                       std::vector<std::string> &responses);
bool chatlog_generator_broken(PromptRunContext &prun_ctx,
                              gpt_params &params,
                              Inference &infer,
                              TextReplacer &replacer,
                              json prompt_runner_conf,
                              json prompt_test,
                              json j_resps,
                              std::vector<std::string> &responses);
bool chatlog_generator_slow(PromptRunContext &prun_ctx,
                            gpt_params &params,
                            Inference &infer,
                            TextReplacer &replacer,
                            json prompt_runner_conf,
                            json prompt_test,
                            json j_resps,
                            std::vector<std::string> &responses);

bool chatlog_generator_broken(PromptRunContext &prun_ctx,
                              gpt_params &params,
                              Inference &infer,
                              TextReplacer &replacer,
                              json prompt_runner_conf,
                              json prompt_test,
                              json j_resps,
                              std::vector<std::string> &responses) {
    StopSequences stop_seq;
    if (prompt_runner_conf.find("stop_sequences") != prompt_runner_conf.end()) {
        stop_seq.set_stop_sequences(prompt_runner_conf["stop_sequences"]);
    }

    Conversation conversation(replacer, stop_seq, prompt_test);
    conversation.load_config(prompt_test["chat"], params.n_predict);

    if (!infer.add_start("char", conversation.char_prompt)) {
        fprintf(stderr, "Couldn't add_start char prompt\n");
        fflush(stderr);
        return false;
    }
    if (!infer.add_start("user", conversation.user_prompt)) {
        fprintf(stderr, "Couldn't add_start user prompt\n");
        fflush(stderr);
        return false;
    }

    if (!infer.append("log", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }
    infer.commit("log");

    bool is_user = true;
    std::string end_reason;

    json raw_chatlog;

    for (int i = 0; i < conversation.chat_turns(); i++) {
        if (infer.current_max_seq_token_count() > 3900) {
            end_reason = "context limit reached: 3900";
            break;
        }

        if (is_user) {
            if (!infer.rebase("log", "user")) {
                fprintf(stderr, "Couldn't rebase to user\n");
                fflush(stderr);
                return false;
            }
        } else {
            if (!infer.rebase("log", "char")) {
                fprintf(stderr, "Couldn't rebase to char\n");
                fflush(stderr);
                return false;
            }
        }

        printf("SEQ[log %d]=%s\n",
               infer.current_max_seq_token_count(),
               infer.get_sequence_text("log").c_str());

        infer.reset_seed(prun_ctx.seed + i);
        infer.reset_sampler("log");

        std::string completion_start =
            conversation.next_completion_fmt(is_user);
        int completion_token_cnt =
            conversation.next_completion_tok_count(is_user);

        if (!infer.complete("log",
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
        std::string completion = infer.get_recently_added_tokens_str("log");
        raw_chatlog.push_back(completion);
        completion = completion.substr(completion_start.size());
        infer.rewind("log");

        std::string log_entry =
            conversation.append_raw_chat_response(is_user, completion);

        printf("> %s", log_entry.c_str());
        if (completion.size() == 0) {
            end_reason = "empty response";
            break;
        }
        if (!infer.append("log", log_entry)) {
            fprintf(stderr, "Couldn't append chatlog\n");
            fflush(stderr);
            return false;
        }
        infer.commit("log");

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
    prompt_collection["raw_chatlog"] = raw_chatlog;
    prompt_collection["max_token_count"] = infer.current_max_seq_token_count();
    prompt_collection["end_reason"] = end_reason;

    j_resps.push_back(make_response(responses,
                                    prun_ctx,
                                    infer.get_sequence_text_no_prev("log"),
                                    infer.get_sequence_token_count("log"),
                                    prompt_collection));

    return true;
}

bool chatlog_generator(PromptRunContext &prun_ctx,
                       gpt_params &params,
                       Inference &infer,
                       TextReplacer &replacer,
                       json prompt_runner_conf,
                       json prompt_test,
                       json j_resps,
                       std::vector<std::string> &responses) {
    StopSequences stop_seq;
    if (prompt_runner_conf.find("stop_sequences") != prompt_runner_conf.end()) {
        stop_seq.set_stop_sequences(prompt_runner_conf["stop_sequences"]);
    }

    Conversation conversation(replacer, stop_seq, prompt_test);
    conversation.load_config(prompt_test["chat"], params.n_predict);

    if (!infer.add_start("char", conversation.char_prompt)) {
        fprintf(stderr, "Couldn't add_start char prompt\n");
        fflush(stderr);
        return false;
    }
    if (!infer.add_start("user", conversation.user_prompt)) {
        fprintf(stderr, "Couldn't add_start user prompt\n");
        fflush(stderr);
        return false;
    }

    if (!infer.append("user", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }

    if (!infer.append("char", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }

    infer.commit("user");
    infer.commit("char");

    bool is_user = true;
    std::string end_reason;

    json raw_chatlog;

    for (int i = 0; i < conversation.chat_turns(); i++) {
        if (infer.current_max_seq_token_count() > 3900) {
            end_reason = "context limit reached: 3900";
            break;
        }

        std::string sequence_name = is_user ? "user" : "char";

        printf("SEQ[log %d]=%s\n",
               infer.current_max_seq_token_count(),
               infer.get_sequence_text(sequence_name).c_str());

        infer.reset_seed(prun_ctx.seed + i);
        infer.reset_sampler(sequence_name);

        std::string completion_start =
            conversation.next_completion_fmt(is_user);
        int completion_token_cnt =
            conversation.next_completion_tok_count(is_user);

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
        std::string completion =
            infer.get_recently_added_tokens_str(sequence_name);
        raw_chatlog.push_back(completion);
        completion = completion.substr(completion_start.size());
        infer.rewind(sequence_name);

        std::string log_entry =
            conversation.append_raw_chat_response(is_user, completion);

        printf("> %s", log_entry.c_str());
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
    prompt_collection["raw_chatlog"] = raw_chatlog;
    prompt_collection["max_token_count"] = infer.current_max_seq_token_count();
    prompt_collection["end_reason"] = end_reason;

    j_resps.push_back(make_response(responses,
                                    prun_ctx,
                                    infer.get_sequence_text_no_prev("log"),
                                    infer.get_sequence_token_count("log"),
                                    prompt_collection));

    return true;
}

bool chatlog_generator_slow(PromptRunContext &prun_ctx,
                            gpt_params &params,
                            Inference &infer,
                            TextReplacer &replacer,
                            json prompt_runner_conf,
                            json prompt_test,
                            json j_resps,
                            std::vector<std::string> &responses) {
    StopSequences stop_seq;
    if (prompt_runner_conf.find("stop_sequences") != prompt_runner_conf.end()) {
        stop_seq.set_stop_sequences(prompt_runner_conf["stop_sequences"]);
    }

    Conversation conversation(replacer, stop_seq, prompt_test);
    conversation.load_config(prompt_test["chat"], params.n_predict);

    if (!infer.add_start("char", conversation.char_prompt)) {
        fprintf(stderr, "Couldn't add_start char prompt\n");
        fflush(stderr);
        return false;
    }
    if (!infer.add_start("user", conversation.user_prompt)) {
        fprintf(stderr, "Couldn't add_start user prompt\n");
        fflush(stderr);
        return false;
    }

    if (!infer.append("log", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }

    bool is_user = true;
    std::string end_reason;

    json raw_chatlog;

    for (int i = 0; i < conversation.chat_turns(); i++) {
        printf("SEQ[log %d]=%s\n",
               infer.current_max_seq_token_count(),
               infer.get_sequence_text("base_prompt").c_str());

        if (infer.current_max_seq_token_count() > 3900) {
            end_reason = "context limit reached: 3900";
            break;
        }

        infer.clear();
        if (is_user) {
            infer.add_start("base_prompt", conversation.user_prompt);
            infer.append("base_prompt", conversation.chatlog_text());

        } else {
            infer.add_start("base_prompt", conversation.char_prompt);
            infer.append("base_prompt", conversation.chatlog_text());
        }
        infer.commit("base_prompt");

        infer.reset_seed(prun_ctx.seed + i);
        infer.reset_sampler("base_prompt");

        std::string completion_start =
            conversation.next_completion_fmt(is_user);
        int completion_token_cnt =
            conversation.next_completion_tok_count(is_user);

        if (!infer.complete("base_prompt",
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
        std::string completion =
            infer.get_recently_added_tokens_str("base_prompt");
        printf("COMPLE[%s]\n", completion.c_str());
        raw_chatlog.push_back(completion);
        completion = completion.substr(completion_start.size());

        std::string log_entry =
            conversation.append_raw_chat_response(is_user, completion);

        printf("> %s", log_entry.c_str());
        if (completion.size() == 0) {
            end_reason = "empty response";
            break;
        }

        is_user = !is_user;
    }

    llama_sampling_params &sparams = params.sparams;

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    conversation.log_chatlog_to_file(
        prun_ctx.seed, model_file, prompt_test, sparams);

    json prompt_collection;
    prompt_collection["char"] = conversation.char_prompt;
    prompt_collection["user"] = conversation.user_prompt;
    prompt_collection["chatlog"] = conversation.chatlog_as_json();
    prompt_collection["raw_chatlog"] = raw_chatlog;
    prompt_collection["max_token_count"] = infer.current_max_seq_token_count();
    prompt_collection["end_reason"] = end_reason;

    j_resps.push_back(
        make_response(responses,
                      prun_ctx,
                      infer.get_sequence_text_no_prev("base_prompt"),
                      infer.get_sequence_token_count("base_prompt"),
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

    fprintf(
        stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

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
    std::vector<std::string> responses;
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
    prun_ctx.total_tests = seeds.size() * temps.size() * test_count;

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

        for (auto &temp : temps) {
            sparams.temp = temp;
            prun_ctx.temp = temp;

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

                if (((int)embd_inp.size() + (int)params.n_predict) >
                    n_ctx - 4) {
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
                                           j_resps,
                                           responses)) {
                        return 1;
                    }

                } else if (prompt_test.find("chat") != prompt_test.end()) {
                }

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

    std::string responses_json_dump =
        j_resps.dump(2, ' ', false, json::error_handler_t::replace);
    printf("%s\n", responses_json_dump.c_str());
    fflush(stdout);

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
