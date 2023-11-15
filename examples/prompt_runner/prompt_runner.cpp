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

std::string trim_generated_chat_response(std::string gen) {
    // Strip extra newlines:
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
    rtrim_nl(gen);
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

            if (tokens.size() < recent_add_tokens) {
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
        ctx_sampling = llama_sampling_init(sparams);
    }

    void reset_seed(int seed) {
        // printf("RESET SEED %d\n", seed);
        llama_set_rng_seed(*g_ctx, seed);
    }

    ~Inference() { llama_sampling_free(ctx_sampling); }

    bool load_tokens(bool is_bos,
                     const std::string &text,
                     int seq_id,
                     int p0,
                     std::vector<llama_token> &out_tokens,
                     int &p1) {
        if (text.size() == 0)
            return true;

        const bool add_bos =
            is_bos && (llama_vocab_type(*g_model) == LLAMA_VOCAB_TYPE_SPM);
        auto tokens = ::llama_tokenize(*g_ctx, text.c_str(), add_bos, true);

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

        int offs = dest->p1 - src->p0;
        llama_kv_cache_seq_shift(*g_ctx, src->seq_id, src->p0, src->p1, offs);
        src->p0 += offs;
        src->p1 += offs;
        src->prev_name = new_prev;

        llama_kv_cache_seq_cp(
            *g_ctx, src->seq_id, dest->seq_id, src->p0, src->p1);
        llama_kv_cache_seq_rm(*g_ctx, src->seq_id, src->p0, src->p1);

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
        printf("complete sequence: seq_id=%d p0=%d, p1=%d\n", seq->seq_id, seq->p0, seq->p1);

        auto tokens = ::llama_tokenize(*g_ctx, text.c_str(), false, true);
        auto batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            bool is_last = (i + 1 == tokens.size());
            //d// printf("ADDBATCH %d (is_last=%d) = %d\n", seq->p1, is_last, tokens[i]);
            llama_batch_add(
                batch, tokens[i], seq->p1, {seq->seq_id}, is_last);
            seq->add_token(tokens[i]);
            llama_sampling_accept(ctx_sampling, *g_ctx, tokens[i], true);
//            printf("accept %d\n", tokens[i]);
        }

        int sample_idx = tokens.size() - 1;

        if (llama_decode(*g_ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);

        //d// printf("COMPLETE SEQUENCE p0=%d, p1=%d, size=%d\n", seq->p0, seq->p1, seq->tokens.size());

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

        if (seq->recent_add_tokens > 0) {
            seq->rewind();
        }

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

        //d// printf("SEQUENCE [%s] seq_id=%d\n", name.c_str(), s->seq_id);

        s->name = name;

        std::vector<llama_token> tokens;
        int new_p1 = 0;
        bool ok = load_tokens(false, text, s->seq_id, s->p1, tokens, new_p1);
        for (auto tok : tokens) {
            s->add_token(tok);
        }
        printf("load tokens seq_id=%d new_p1=%d, seq->p1=%d\n", s->seq_id, new_p1, s->p1);
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

struct TokenVec {
    llama_pos p0;
    llama_pos p1;
    std::vector<llama_token> tokens;
    bool decoded;

    TokenVec() : p0(0), p1(0), decoded(false) {}
    TokenVec(const std::string &prompt) : p0(0), p1(0), decoded(false) {
        load_as_mid(prompt);
    }

    TokenVec spawn_continuation() {
        TokenVec tv;
        if (tokens.size() > 0) {
            tv.tokens.push_back(tokens.back());
        }
        tv.p0 = p1;
        tv.p1 = p1;
        return tv;
    }

    void append(llama_token id) { tokens.push_back(id); }
    void append(TokenVec &other_vec) {
        for (auto tok : other_vec.tokens) {
            tokens.push_back(tok);
        }
        p1 = other_vec.p1;
    }

    std::string to_string() {
        std::string p;
        for (auto tok : tokens) {
            p += llama_token_to_piece(*g_ctx, tok);
        }
        return p;
    }

    void print() {
        std::string p;
        for (auto tok : tokens) {
            p += llama_token_to_piece(*g_ctx, tok);
        }
        printf("### tokens=[%s]\n", p.c_str());
    }

    void print_last() {
        std::string p = llama_token_to_piece(*g_ctx, get_last());
        printf("### last_token=[%s]\n", p.c_str());
    }

    llama_token get_last() {
        if (tokens.size() > 0) {
            return tokens[tokens.size() - 1];
        }
        return 0;
    }

    void load_as_bos(const std::string &prompt) {
        const bool add_bos = llama_vocab_type(*g_model) == LLAMA_VOCAB_TYPE_SPM;
        tokens = ::llama_tokenize(*g_ctx, prompt.c_str(), add_bos, true);
        // d// printf("TOKENIZE BOS[%s]\n", prompt.c_str());
    }

    void load_as_mid(const std::string &prompt) {
        tokens = ::llama_tokenize(*g_ctx, prompt.c_str(), false, true);
        // d// printf("TOKENIZE MID[%s]\n", prompt.c_str());
    }

    size_t size() { return tokens.size(); }
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

    json char_cfg;
    json user_cfg;

    std::vector<std::string> chatlog;

    json test_replacements;

    Conversation(TextReplacer repl, StopSequences ss, json test_replacement)
        : replacer(repl), stop_seq(ss), test_replacements(test_replacement) {}

    bool load_config(json chat) {
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
        printf("FOFO\n");

        user_prompt = replacer.apply_replacements(test_replacements,
                                                  user_cfg.value("prompt", ""));
        char_prompt = replacer.apply_replacements(test_replacements,
                                                  char_cfg.value("prompt", ""));
        user_log_fmt = user_cfg.value("log_fmt", "");
        char_log_fmt = char_cfg.value("log_fmt", "");
        user_next_fmt = replacer.apply_replacements(
            test_replacements, user_cfg.value("next_fmt", ""));
        char_next_fmt = replacer.apply_replacements(
            test_replacements, char_cfg.value("next_fmt", ""));
        printf("FOFO %s\n", user_prompt.c_str());

        std::string char_log_init = chat.value("char_log_init", "");
        printf("FOFO\n");
        if (char_log_init.size() > 0) {
            replacer.add_extra("<RESPONSE>", char_log_init);
            std::string entry =
                replacer.apply_replacements(test_replacements, char_log_fmt);
            chatlog.push_back(entry);
        }

        return true;
    }

    std::string next_completion_fmt(bool is_user) {
        return is_user ? user_next_fmt : char_next_fmt;
    }

    std::string append_raw_chat_response(bool is_user,
                                         const std::string &resp) {
        std::string cleaned;
        stop_seq.trim_stop_sequence(resp, cleaned);
        //        printf("PREAPPENDLOG:[%s]\n", cleaned.c_str());
        cleaned = trim_generated_chat_response(cleaned);

        trim_nl(cleaned, " \r\n");
        replacer.add_extra("<RESPONSE>", cleaned);
        std::string entry = replacer.apply_replacements(
            test_replacements, is_user ? user_log_fmt : char_log_fmt);
        chatlog.push_back(entry);
        printf("APPENDLOG:[%s]\n", chatlog.back().c_str());
        return entry;
    }

    std::string chatlog_text() { return concatl(chatlog); }

    std::string chatlog_text_ext() { return concatl(chatlog, true); }

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

struct Decoder {
    struct llama_sampling_context *ctx_sampling;

    Decoder(const llama_sampling_params &sparams) {
        ctx_sampling = llama_sampling_init(sparams);
    }

    void init() { llama_kv_cache_clear(*g_ctx); }

    void reset_seed(int seed) {
        //        printf("RESET SEED %d\n", seed);
        llama_set_rng_seed(*g_ctx, seed);
    }

    void refeed_tokens_for_sampling(TokenVec &tokens) {
        printf("REFEED[p0=%d,p1=%d,size=%ld][%s]\n",
               tokens.p0,
               tokens.p1,
               tokens.size(),
               tokens.to_string().c_str());
        llama_sampling_reset(ctx_sampling);
        for (auto tok : tokens.tokens) {
            llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);
        }
    }

    llama_token sample(int idx) {
        // printf("SAMPLE IDX=%d\n", idx);
        const llama_token id =
            llama_sampling_sample(ctx_sampling, *g_ctx, NULL, idx);
        return id;
    }

    bool decode_to_cache(TokenVec &tokens, llama_pos p0, llama_seq_id seq_id) {
        llama_batch batch;

        batch = llama_batch_init(tokens.size(), 0, 1);
        int i = 0;
        // tokens.print();
        printf("### batch seq_id=%d, p0=%d to p1=%ld\n",
               seq_id,
               p0,
               p0 + tokens.size());
        for (auto tok : tokens.tokens) {
            llama_batch_add(batch, tok, p0 + i, {seq_id}, false);
            llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);
            i++;
        }

        bool ok = false;
        if (llama_decode(*g_ctx, batch) != 0) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            ok = false;
        } else {
            tokens.p0 = p0;
            tokens.p1 = p0 + i;
            ok = true;
        }

        llama_batch_free(batch);

        return ok;
    }

    bool decode(TokenVec &tokens,
                bool only_last_token,
                llama_pos p0,
                llama_seq_id seq_id) {
        llama_batch batch;

        if (only_last_token) {
            batch = llama_batch_init(1, 0, 1);
            // d// printf("### batch seq_id=%d, p0=%d\n", seq_id, p0);
            // d// tokens.print_last();
            llama_batch_add(batch, tokens.get_last(), p0, {seq_id}, true);
            llama_sampling_accept(
                ctx_sampling, *g_ctx, tokens.get_last(), true);
            tokens.p0 = p0;
            tokens.p1 = p0 + 1;

        } else {
            batch = llama_batch_init(tokens.size(), 0, 1);
            int i = 0;
            //            tokens.print();
            // printf("### batch seq_id=%d, p0=%d to p1=%ld\n", seq_id,
            //        p0, p0 + tokens.size());
            for (auto tok : tokens.tokens) {
                llama_batch_add(batch, tok, p0 + i, {seq_id}, false);
                llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);
                i++;
            }
            batch.logits[batch.n_tokens - 1] = true;

            tokens.p0 = p0;
            tokens.p1 = p0 + i;
        }

        bool ok = false;
        if (llama_decode(*g_ctx, batch) != 0) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
        } else {
            tokens.decoded = true;
            ok = true;

            tokens.append(sample(batch.n_tokens - 1));
        }

        llama_batch_free(batch);

        return ok;
    }
};

int prompt_piece_seq_id = 0;

struct SystemPrompt {
    llama_pos p0;
    llama_seq_id seq_id;
    TokenVec tokens;

    SystemPrompt(const std::string &prompt) : p0(0) {
        seq_id = prompt_piece_seq_id++;
        tokens.load_as_bos(prompt);
    }

    bool decode(Decoder &decoder) {
        return decoder.decode_to_cache(tokens, p0, seq_id);
    }

    std::string to_string() { return tokens.to_string(); }

    llama_pos get_p1() { return tokens.p1; }
};

struct PromptPiece {
    llama_pos p0;
    llama_seq_id seq_id;
    TokenVec tokens;

    void load_as_mid(const std::string &prompt) {
        p0 = 0;
        seq_id = 0;
        tokens.load_as_mid(prompt);
    }

    llama_pos get_p0() { return tokens.p0; }
    llama_pos get_p1() { return tokens.p1; }

    void shift_to(llama_pos new_p0, llama_seq_id new_seq_id) {
        llama_pos offs = new_p0 - tokens.p0;

        llama_kv_cache_seq_cp(*g_ctx, seq_id, new_seq_id, tokens.p0, tokens.p1);
        llama_kv_cache_seq_shift(
            *g_ctx, new_seq_id, tokens.p0, tokens.p1, offs);

        seq_id = new_seq_id;
        tokens.p0 = tokens.p0 + offs;
        tokens.p1 = tokens.p1 + offs;
        p0 = tokens.p0;
    }

    bool append(Decoder &decoder, TokenVec &new_tokens) {
        if (!decoder.decode_to_cache(new_tokens, tokens.p1, seq_id)) {
            return false;
        }
        printf("appended tokens p0=%d, p1=%d\n", new_tokens.p0, new_tokens.p1);

        tokens.append(new_tokens);

        return true;
    }

    bool decode_to_cache(Decoder &decoder, llama_pos p0, llama_seq_id sid) {
        seq_id = sid;

        if (!decoder.decode_to_cache(tokens, p0, seq_id)) {
            return false;
        }

        return true;
    }

    bool complete(Decoder &decoder,
                  StopSequences &stop_seq,
                  const std::string &next_piece,
                  int n_tokens,
                  std::string &output) {
        int pre_usage = llama_kv_cache_usage(**g_ctx);

        decoder.refeed_tokens_for_sampling(tokens);

        TokenVec leader_tokens(next_piece);

        //  printf("START! [%s]\n", leader_tokens.to_string().c_str());
        if (!decoder.decode(leader_tokens, false, tokens.p1, seq_id)) {
            return false;
        }
        printf("LEADER[p0=%d,p1=%d,size=%ld][%s]\n",
               leader_tokens.p0,
               leader_tokens.p1,
               leader_tokens.size(),
               leader_tokens.to_string().c_str());

        TokenVec new_tokens = leader_tokens.spawn_continuation();

        // printf("MID! [%s]\n", new_tokens.to_string().c_str());
        while (n_tokens > 0) {
            //            printf("NEW[p0=%d,p1=%d][%s]\n", new_tokens.p0,
            //            new_tokens.p1,
            //                   new_tokens.to_string().c_str());
            if (new_tokens.size() > 0 &&
                new_tokens.tokens.back() == llama_token_eos(*g_model)) {
                // printf("got EOS\n");
                break;
            }

            std::string generated = new_tokens.to_string();
            std::string _out;
            if (stop_seq.trim_stop_sequence(generated, _out)) {
                // printf("got STOPSEQ\n");
                break;
            }

            if (!decoder.decode(new_tokens, true, new_tokens.p1, seq_id)) {
                printf("ERROR?!?\n");
                return false;
            }

            n_tokens--;
        }

        // int pre_usage = llama_kv_cache_usage(**g_ctx);

        output = new_tokens.to_string();
        // printf("SETOUT:[%s]\n", output.c_str());

        // tokens.p0 is the beginning of the completion, including the leading
        // prompt new_tokens.p1 is the end of the generated tokens, pointing
        // towards the next token.
        llama_kv_cache_seq_rm(*g_ctx, seq_id, tokens.p1, new_tokens.p1);

        int post_usage = llama_kv_cache_usage(**g_ctx);
        if (post_usage != pre_usage) {
            printf("PREPOST DIFF=%d (%d)\n", pre_usage - post_usage, pre_usage);
        }
        assert(pre_usage == post_usage);
        //  printf("OUTPUT! %d|%d %d,%d => %d [%s]\n", tokens.p1, tokens.p1 +
        //  new_tokens.size(), pre_usage1, pre_usage, post_usage,
        //         output.c_str());

        return true;
    }
};

struct DualPrefixPrompt {
    SystemPrompt prefix1;
    SystemPrompt prefix2;
    bool is_at_prefix1;
    PromptPiece mid_piece;

    DualPrefixPrompt(const std::string &prefix1, const std::string &prefix2)
        : prefix1(prefix1), prefix2(prefix2), is_at_prefix1(true) {}

    void set_mid_piece(const std::string &dyn_prompt_init) {
        mid_piece.load_as_mid(dyn_prompt_init);
    }

    bool init_decode(Decoder &decoder) {
        if (!prefix1.decode(decoder)) return false;
        if (!prefix2.decode(decoder)) return false;

        int p1 = prefix1.get_p1();
        is_at_prefix1 = true;
        return mid_piece.decode_to_cache(decoder, p1, prefix1.seq_id);
    }

    void use_prefix1() {
        printf("USE PREFIX1(p1=%d)[%s]\n",
               prefix1.get_p1(),
               prefix1.tokens.to_string().c_str());
        if (!is_at_prefix1) {
            mid_piece.shift_to(prefix1.get_p1(), prefix1.seq_id);
            is_at_prefix1 = true;
        }
    }

    void use_prefix2() {
        printf("USE PREFIX2(p1=%d)[%s]\n",
               prefix2.get_p1(),
               prefix2.tokens.to_string().c_str());
        if (is_at_prefix1) {
            mid_piece.shift_to(prefix2.get_p1(), prefix2.seq_id);
            is_at_prefix1 = false;
        }
    }

    size_t prompt_length() {
        size_t len = prefix1.tokens.size();
        if (prefix2.tokens.size() > len) len = prefix2.tokens.size();
        len += mid_piece.tokens.size();
        return len;
    }

    bool feed_mid_piece(Decoder &decoder, const std::string &mid) {
        TokenVec new_tokens(mid);
        return mid_piece.append(decoder, new_tokens);
    }

    bool complete(Decoder &decoder,
                  StopSequences &stop_seq,
                  const std::string &add_prompt,
                  int n_tokens,
                  std::string &output) {
        return mid_piece.complete(
            decoder, stop_seq, add_prompt, n_tokens, output);
    }
};

struct PromptProcessor {
    std::vector<llama_token> embd;
    std::vector<llama_token> embd_gen;
    json stop_sequences;
    int n_past;
    int n_batch;
    struct llama_sampling_context *ctx_sampling;
    int last_response_tok_count;
    std::string last_raw_response;
    bool broken_rep_pen;

    PromptProcessor(int batch,
                    struct llama_sampling_context *ctxs,
                    json prompt_runner_conf)
        : n_past(0),
          n_batch(batch),
          ctx_sampling(ctxs),
          last_response_tok_count(0),
          broken_rep_pen(false) {
        init();
        get_stop_sequences(prompt_runner_conf);
    }

    void set_broken_rep_pen() {
        // due to a bug in the ERP3 benchmark run only the prompt
        // was added to repetition penalty. For backward compat we provide this.
        broken_rep_pen = true;
    }

    void get_stop_sequences(json prompt_runner_conf) {
        if (prompt_runner_conf.find("stop_sequences") !=
            prompt_runner_conf.end()) {
            stop_sequences = prompt_runner_conf["stop_sequences"];
        }
    }

    void init() { llama_kv_cache_clear(*g_ctx); }

    void add_tokens(TokenVec &tokens) { add_tokens(tokens.tokens); }

    void add_tokens(const std::vector<llama_token> &tokens) {
        for (auto tok : tokens) {
            embd.push_back(tok);
            llama_sampling_accept(ctx_sampling, *g_ctx, tok, false);
        }
    }

    bool generate_tokens(int n_remain) {
        while (n_remain > 0) {
            if (!this->process_tokens()) {
                return true;
            }

            llama_token id =
                llama_sampling_sample(ctx_sampling, *g_ctx, nullptr);

            this->add_generated(id);
            --n_remain;

            if (this->reached_eos()) {
                break;
            }

            if (this->reached_stop_sequence()) {
                n_remain = 0;
                break;
            }
        }

        return true;
    }

    void add_token(llama_token t) {
        // push the prompt in the sampling context in order
        // to apply repetition penalties later for the
        // prompt, we don't apply grammar rules
        embd.push_back(t);
        if (!broken_rep_pen) {
            llama_sampling_accept(ctx_sampling, *g_ctx, t, false);
        }
    }

    void add_generated(llama_token t) {
        add_token(t);
        embd_gen.push_back(t);
    }

    bool reached_eos() {
        if (!embd.empty() && embd.back() == llama_token_eos(*g_model)) {
            // fprintf(stderr, " [end of text]\n");
            return true;
        }

        return false;
    }

    int get_last_response_token_count() { return last_response_tok_count; }

    std::string get_response(bool without_stop_seq = false,
                             bool trim_trailing = false) {
        last_response_tok_count = 0;
        std::string gen = "";
        for (auto id : embd_gen) {
            last_response_tok_count += 1;
            gen += llama_token_to_piece(*g_ctx, id);
        }
        last_raw_response = gen;

        rtrim_nl(gen);

        if (without_stop_seq) {
            for (const auto &matched : stop_sequences) {
                size_t stop_pos = gen.find(matched);
                if (stop_pos != std::string::npos) {
                    gen = gen.substr(0, stop_pos);
                    break;
                }
            }
        }

        if (trim_trailing) {
            gen = std::regex_replace(
                gen, std::regex("\n\n\n*", std::regex::extended), "\n");
            gen = std::regex_replace(
                gen, std::regex("\\.\\.\\.*", std::regex::extended), "");
            gen = std::regex_replace(
                gen, std::regex("\\*\\*\\**", std::regex::extended), "");
            gen =
                std::regex_replace(gen,
                                   std::regex("(.*[.!?*\")}`$])[^.!?*\")}`$]*",
                                              std::regex::extended),
                                   "$1");
        }

        return gen;
    }

    bool reached_stop_sequence() {
        std::string generated = "";
        for (auto id : embd_gen) {
            generated += llama_token_to_piece(*g_ctx, id);
        }

        for (const auto &matched : stop_sequences) {
            if (generated.find(matched) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    bool process_tokens() {
        const int n_ctx = llama_n_ctx(*g_ctx);

        if (embd.size() <= 0) {
            return true;
        }

        // Note: n_ctx - 4 here is to match the logic for
        // commandline prompt handling via
        // --prompt or --file which uses the same value.
        auto max_embd_size = n_ctx - 4;
        // Ensure the input doesn't exceed the context size by
        // truncating embd if necessary.
        if ((int)embd.size() > max_embd_size) {
            auto skipped_tokens = embd.size() - max_embd_size;
            printf("<<input too long: skipped %zu token%s>>",
                   skipped_tokens,
                   skipped_tokens != 1 ? "s" : "");
            fflush(stdout);
            embd.resize(max_embd_size);
        }

        // evaluate tokens in batches
        // embd is typically prepared beforehand to fit within a
        // batch, but not always
        for (int i = 0; i < (int)embd.size(); i += n_batch) {
            int n_eval = (int)embd.size() - i;
            if (n_eval > n_batch) {
                n_eval = n_batch;
            }
            if (llama_decode(
                    *g_ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return false;
            }
            n_past += n_eval;
        }

        embd.clear();

        return true;
    }
};

void record_token_info(std::vector<std::string> &responses,
                       json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling);

void record_token_info(std::vector<std::string> &responses,
                       json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling) {
    const int n_vocab = llama_n_vocab(*g_model);

    float *logits = llama_get_logits_ith(*g_ctx, 0);

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

                    if (true || params.n_batch == 512) {
                        Inference infer(sparams, params.n_batch);

                        StopSequences stop_seq;
                        if (prompt_runner_conf.find("stop_sequences") !=
                            prompt_runner_conf.end()) {
                            stop_seq.set_stop_sequences(
                                prompt_runner_conf["stop_sequences"]);
                        }

                        Conversation conversation(
                            replacer, stop_seq, prompt_test);
                        conversation.load_config(prompt_test["chat"]);

                        infer.add_start("user", conversation.user_prompt);
                        infer.add_start("char", conversation.char_prompt);
                        infer.append("log", conversation.chatlog_text());
                        infer.commit("log");
                        infer.rebase("log", "user");

                        infer.reset_seed(seed_value);
                        infer.reset_sampler("log");

                        bool is_user = true;
                        infer.complete(
                            "log",
                            conversation.next_completion_fmt(is_user),
                            70,
                            [](int cnt, const std::string &comp) {
//                                if (cnt > 65) {
//                                    printf("C:[%d][%s]\n", cnt, comp.c_str());
//                                }
                                return false;
                            });
                        printf(
                            "Completed[%s]\n",
                            infer.get_recently_added_tokens_str("log").c_str());

                        infer.rewind("log");

                        infer.reset_seed(seed_value);
                        infer.reset_sampler("log");

                        infer.complete(
                            "log",
                            conversation.next_completion_fmt(is_user),
                            70,
                            [](int cnt, const std::string &comp) {
//                                if (cnt > 65) {
//                                    printf("C:[%d][%s]\n", cnt, comp.c_str());
//                                }
                                return false;
                            });
                        printf(
                            "Completed2[%s]\n",
                            infer.get_recently_added_tokens_str("log").c_str());

                        infer.rewind("log");

                        infer.append("log", "Loki");
                        infer.reset_seed(seed_value);
                        infer.reset_sampler("log");

                        infer.complete(
                            "log",
                            ":",//conversation.next_completion_fmt(is_user),
                            70,
                            [](int cnt, const std::string &comp) {
//                                if (cnt > 65) {
//                                    printf("C:[%d][%s]\n", cnt, comp.c_str());
//                                }
                                return false;
                            });
                        printf(
                            "Completed[%s]\n",
                            infer.get_recently_added_tokens_str("log").c_str());

                    } else {
                        StopSequences stop_seq;
                        if (prompt_runner_conf.find("stop_sequences") !=
                            prompt_runner_conf.end()) {
                            stop_seq.set_stop_sequences(
                                prompt_runner_conf["stop_sequences"]);
                        }

                        Conversation conversation(
                            replacer, stop_seq, prompt_test);
                        conversation.load_config(prompt_test["chat"]);

                        DualPrefixPrompt dpp(conversation.user_prompt,
                                             conversation.char_prompt);

                        Decoder decoder(sparams);
                        decoder.init();
                        dpp.set_mid_piece(conversation.chatlog_text());
                        dpp.init_decode(decoder);

                        printf("MID[%s]\n",
                               dpp.mid_piece.tokens.to_string().c_str());

                        std::string out;

                        decoder.reset_seed(seed_value);
                        for (int k = 0; k < 1; k++) {
                            printf(
                                "##############################################"
                                "####"
                                "#######################\n");
                            printf(
                                "#####%7d "
                                "##############################################"
                                "####"
                                "##########\n",
                                k);
                            printf(
                                "##############################################"
                                "####"
                                "#######################\n");
                            bool is_user = k % 2 == 0;
                            if (is_user) {
                                dpp.use_prefix1();
                            } else {
                                dpp.use_prefix2();
                            }

                            dpp.complete(
                                decoder,
                                stop_seq,
                                conversation.next_completion_fmt(is_user),
                                70,
                                out);
                            std::string log_entry =
                                conversation.append_raw_chat_response(is_user,
                                                                      out);
                            dpp.feed_mid_piece(decoder, log_entry);

                            if (dpp.prompt_length() > 3900) {
                                break;
                            }
                        }

                        std::string model_file = params.model.c_str();
                        model_file = model_file.substr(
                            model_file.find_last_of("/\\") + 1);
                        conversation.log_chatlog_to_file(
                            seed_value, model_file, prompt_test, sparams);
                    }

                    //                    decoder.reset_seed(seed_value);
                    //                    dpp.complete(decoder, stop_seq,
                    //                                 conversation.next_completion_fmt(false),
                    //                                 70, out);
                    //                    printf("GEN1[%s]\n", out.c_str());
                    //                    decoder.reset_seed(seed_value);
                    //                    dpp.complete(decoder, stop_seq,
                    //                                 conversation.next_completion_fmt(false),
                    //                                 70, out);
                    //                    printf("GEN2[%s]\n", out.c_str());

                    //                    dpp.complete(decoder, stop_seq,
                    //                    "Loki:", 70, out);
                    //
                    //                    printf("OUTPUT[%s]\n", out.c_str());
                    //                    decoder.reset_seed(seed_value);
                    //
                    //                    printf("### rewinding kv cache...\n");
                    //
                    //                    dpp.complete(decoder, stop_seq,
                    //                    "Loki:", 70, out);
                    //                    printf("OUTPUT[%s]\n", out.c_str());
                    //
                    //                    for (int i = 0; i < 100; i++) {
                    //                        if (i % 2 == 0) {
                    //                            if (i > 0) {
                    //                                dpp.swap_prefix_and_complete(decoder,
                    //                                                             "\nLoki: ", 70);
                    //                            } else {
                    //                                dpp.swap_prefix_and_complete(decoder,
                    //                                                             "Loki: ", 70);
                    //                            }
                    //                        } else {
                    //                            if (i > 1) {
                    //                                dpp.swap_prefix_and_complete(decoder,
                    //                                                             "\nAria: ", 70);
                    //                            } else {
                    //                                dpp.swap_prefix_and_complete(decoder,
                    //                                                             "Aria: ", 70);
                    //                            }
                    //                        }
                    //                    }

                    //                    int chat_turns = chat.value("turns",
                    //                    3);
                    //
                    //
                    //                    std::string char_log_init =
                    //                    chat.value("char_log_init", ""); if
                    //                    (char_log_init.size() > 0) {
                    //                        replacer.add_extra("<RESPONSE>",
                    //                        char_log_init); std::string
                    //                        log_fmt =
                    //                            char_prompt.value("log_fmt",
                    //                            "<RESPONSE>");
                    //                        char_log_init =
                    //                            replacer.apply_replacements(prompt_test,
                    //                            log_fmt);
                    //                        chatlog.push_back(char_log_init);
                    //                    }
                    //
                    //                    bool is_char_turn = false;
                    //                    int gen_token_count_sum = 0;
                    //                    json raw_chatlog;
                    //                    bool ended_with_empty = false;
                    //                    bool repeated_itself = false;
                    //                    std::string repeated_part;
                    //                    std::string repeated_log_entry;
                    //                    std::string repeated_current_entry;
                    //
                    //                    json last_char_prompt;
                    //                    json last_user_prompt;
                    //
                    //                    for (int turn_i = 0; turn_i <
                    //                    chat_turns; turn_i++) {
                    //                        std::string whose_response =
                    //                        "user"; json chat_prompt =
                    //                        user_prompt; if (is_char_turn) {
                    //                            whose_response = "char";
                    //                            chat_prompt = char_prompt;
                    //                        }
                    //
                    //                        int n_remain =
                    //                        chat_prompt.value("n_gen", 100);
                    //                        std::string prompt =
                    //                            chat_prompt.value("prompt",
                    //                            "NO CHAT PROMPT");
                    //                        std::string log_fmt =
                    //                            chat_prompt.value("log_fmt",
                    //                            "<RESPONSE>");
                    //
                    //                        printf("##################################\n");
                    //                        std::string logstr =
                    //                        concatl(chatlog);
                    //                        printf("LOG:\n%s",
                    //                        logstr.c_str()); fflush(stdout);
                    //
                    //                        replacer.add_extra("<CHATLOG>",
                    //                        logstr); prompt =
                    //                            replacer.apply_replacements(prompt_test,
                    //                            prompt);
                    //                        rtrim_nl(prompt);
                    //
                    //                        // d// printf("FOOO5[%s]\n",
                    //                        prompt.c_str());
                    //                        std::vector<llama_token>
                    //                        chat_embd_inp; chat_embd_inp =
                    //                        ::llama_tokenize(ctx,
                    //                        prompt.c_str(),
                    //                                                         add_bos, true);
                    //                        printf("########( PLEN: %ld
                    //                        )##############\n",
                    //                               chat_embd_inp.size());
                    //
                    //                        json prompt_info;
                    //                        prompt_info["tokens"] =
                    //                        (int)chat_embd_inp.size();
                    //                        prompt_info["prompt"] = prompt;
                    //
                    //                        if (is_char_turn) {
                    //                            last_char_prompt =
                    //                            prompt_info;
                    //                        } else {
                    //                            last_user_prompt =
                    //                            prompt_info;
                    //                        }
                    //                        is_char_turn = !is_char_turn;
                    //
                    //                        prun_ctx.prompt_token_cnt =
                    //                        chat_embd_inp.size();
                    //
                    //                        struct llama_sampling_context
                    //                        *ctx_sampling =
                    //                            llama_sampling_init(sparams);
                    //
                    //                        PromptProcessor
                    //                        proc(params.n_batch, ctx_sampling,
                    //                                             prompt_runner_conf);
                    //
                    //                        if
                    //                        (prompt_test.value("broken_rep_pen",
                    //                        false)) {
                    //                            proc.set_broken_rep_pen();
                    //                        }
                    //
                    //                        proc.add_tokens(chat_embd_inp);
                    //                        proc.generate_tokens(n_remain);
                    //
                    //                        std::string gen =
                    //                        proc.get_response(true, true); int
                    //                        gen_tok_cnt =
                    //                        proc.get_last_response_token_count();
                    //
                    //                        json logentry;
                    //                        logentry.push_back(whose_response);
                    //                        logentry.push_back(gen_tok_cnt);
                    //                        logentry.push_back(proc.last_raw_response);
                    //                        raw_chatlog.push_back(logentry);
                    //
                    //                        gen_token_count_sum +=
                    //                        gen_tok_cnt; trim_nl(gen, "
                    //                        \r\n"); if (gen.size() == 0) {
                    //                            ended_with_empty = true;
                    //                            printf(
                    //                                "EMPTY "
                    //                                "RESPONSE!\n################################\n%"
                    //                                "s\n*******************************************"
                    //                                "************\n",
                    //                                prompt.c_str());
                    //                        }
                    //                        replacer.add_extra("<RESPONSE>",
                    //                        gen); std::string new_log =
                    //                            replacer.apply_replacements(prompt_test,
                    //                            log_fmt);
                    //                        fflush(stdout);
                    //                        chatlog.push_back(new_log);
                    //                        if
                    //                        (chatlog_has_repetitions(chatlog,
                    //                        repeated_part,
                    //                                                    repeated_log_entry))
                    //                                                    {
                    //                            repeated_current_entry =
                    //                            new_log; repeated_itself =
                    //                            true; break;
                    //                        }
                    //
                    //                        llama_sampling_free(ctx_sampling);
                    //
                    //                        if (gen.size() == 0) {
                    //                            break;
                    //                        }
                    //                    }
                    //
                    //                    json prompt_collection;
                    //                    prompt_collection["char"] =
                    //                    last_char_prompt;
                    //                    prompt_collection["user"] =
                    //                    last_user_prompt;
                    //                    prompt_collection["raw_chatlog"] =
                    //                    raw_chatlog; json end_reason;
                    //                    end_reason["empty_response"] =
                    //                    ended_with_empty;
                    //                    end_reason["repeated_response"] =
                    //                    repeated_itself; if (repeated_itself)
                    //                    {
                    //                        end_reason["repeated_part"] =
                    //                        repeated_part;
                    //                        end_reason["repeated_log_entry"] =
                    //                        repeated_log_entry;
                    //                        end_reason["repeated_current_entry"]
                    //                        =
                    //                            repeated_current_entry;
                    //                    }
                    //                    prompt_collection["end_reason"] =
                    //                    end_reason;
                    //
                    //                    j_resps.push_back(
                    //                        make_response(responses, prun_ctx,
                    //                        concatl(chatlog),
                    //                                      gen_token_count_sum,
                    //                                      prompt_collection));

                } else {
                    struct llama_sampling_context *ctx_sampling =
                        llama_sampling_init(sparams);

                    PromptProcessor proc(
                        params.n_batch, ctx_sampling, prompt_runner_conf);
                    if (prompt_test.value("broken_rep_pen", false)) {
                        proc.set_broken_rep_pen();
                    }
                    proc.add_tokens(embd_inp);

                    int n_remain = params.n_predict;
                    while (n_remain > 0) {
                        if (!proc.process_tokens()) {
                            return 1;
                        }

                        if (record_next_token_info) {
                            record_token_info(
                                responses, j_tok_resps, prun_ctx, ctx_sampling);
                        }

                        llama_token id = 0;
                        if (sample_seeds.size() > 0) {
                            id = generate_sample_seeded(responses,
                                                        j_resps,
                                                        sample_seeds,
                                                        prun_ctx,
                                                        ctx_sampling);
                        } else {
                            id = llama_sampling_sample(
                                ctx_sampling, ctx, nullptr);
                        }

                        proc.add_generated(id);
                        --n_remain;

                        if (proc.reached_eos()) {
                            break;
                        }

                        if (proc.reached_stop_sequence()) {
                            n_remain = 0;
                            break;
                        }
                    }

                    std::string gen = proc.get_response();
                    int gen_tok_cnt = proc.get_last_response_token_count();

                    j_resps.push_back(make_response(
                        responses, prun_ctx, gen, gen_tok_cnt, json()));

                    llama_sampling_free(ctx_sampling);
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
