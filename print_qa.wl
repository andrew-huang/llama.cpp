!std_opts = $[
    $["-f", :FILE, "The input JSON file, if none provided, the lastest named 'result_*.json' in the current dir is taken."],
    $["-m", "--min-p", "Prints the min-p tokens"],
    $["--file-match", :MATCH, "substring of the filenames that are considered"],
    $["--test-id", :TEST_ID, "Selects only the results with this test ID (you can give an index, to select the first, second, ... test-id)"],
    $["--seed", "-s", :SEED, "Selects only the results with this seed ID (you can give an index below 99, to select the (i-1)th seed)"],
    $["--category", "-c", :CATEGORY, "Selects only the nodes with this category (you can give an index, to select the (i-1)th category)"],
    $["--topic", :TOPIC, "Selects only the nodes with this topic (match by substring)."],
    $["--tree"],
    $["-t", :TOP_N, $o(1)],
];
!cfg = std:app:simple_cli
    "print_qa" "1.0.0"
    "Printing a prompt_runner/query_runner response file's IQ question results."
    :stat => $[*std_opts]
    :raw_print => $[*std_opts]
    :print => $[*std_opts]
    :print_full => $[*std_opts]
    :rank => $[*std_opts]
    :list => $[
        $["-t", :TOP_N, $o(1)],
        $["-f", :FILE, "The input JSON file, if none provided, the lastest named 'result_*.json' in the current dir is taken."],
    ];

!category_weights = ${
    iq4_questions = 0.5,
    iq4_coherency_stmt_colors = 0.25,
    iq4_coherency_topic_colors = 0.25,
};

!CATEGORY_LIST = map {|| _1 } category_weights;
std:sort { std:cmp:str:asc _ _1 } CATEGORY_LIST;


!files = if std:str:len[cfg.f] > 0 {
    $[cfg.f]
} {
    !files = $[];
    std:fs:read_dir "." {
        if _.name &> $r/$^result_*\.json$$/ {
            if is_some[cfg.file-match] {
                if is_some[0 => cfg.file-match _.path] {
                    files +> _.mtime => _.path;
                };
            } {
                files +> _.mtime => _.path;
            };
        };
        $f
    };

    std:sort { std:cmp:num:desc _.0 _1.0 } files;
    std:take cfg.t ~ map { _.1 } files
};

!parse_probs = {!(s) = @;
    !probs = $[];
    while len[s] > 0 {
        !(l, rest) = ":" => 2 s;
        .l = int l;
        !token_str = 0 => int[l] rest;
        .rest = l => -1 rest;
        !(prob, r) = ";" => 2 rest;
        .prob = float ~ std:str:trim prob;
        std:push probs token_str => float[prob];
        .s = r;
    };
    probs
};

!get_word = {!(probs) = @;
    !word = "";
    !prob = 1.0;
    iter p probs {
        if p.0 &> $r/$^$+[a-zA-Z]$$/ {
            .word +>= p.0;
            .prob *= p.1;
        } {
            break[];
        };
    };
    word => prob
};

!wrap_text_char = {!(text, maxlen) = @;
    if text &> $r/$^(^$+[^\:]\:)(^*)$$/ {
        !charname = $\.1;
        if std:str:len[charname] > 20 {
            return text;
        };

        !text = $\.2;
        !out = "";
        !pad = $@s range 0 std:str:len[charname] 1 {|| $+ " "};
        !line = charname;

        !seq = std:str:nlp:match_prefix_and_split_sequences $[] text;

        iter w seq {
            !word = w.1;
            .line +>= word;
            if w.1 &> $r/$^$+$s$$/ &and std:str:len[line] > maxlen {
                .out +>= (if len[out] > 0 { pad } { "" }) std:str:trim[line];
                .out +>= "\n";
                .line = "";
            };
        };
        if std:str:len[line] > 0 {
            .out +>= (if len[out] > 0 { pad } { "" }) std:str:trim[line];
        };
        return out;
    };

    text
};

!stat_prompts = {|1<4| !(prompts, categories, print, print_full) = @;
    is_none[categories] { .categories = ${} };

    !last = $n;
    iter p prompts {
#        std:displayln ~ std:ser:json p;
        if is_some[cfg.category] {
            if cfg.category &> $r/$^$+[0-9]$$/ {
                if is_none[p.payload] &or p.payload.category != CATEGORY_LIST.(int cfg.category) {
                    .last = p;
                    next[];
                };
            } {
                if is_none[p.payload] &or p.payload.category != cfg.test-id {
                    .last = p;
                    next[];
                };
            };
        };

        if is_some[cfg.topic] {
            if is_none[p.payload] &or is_none[0 => cfg.topic p.payload.topic] {
                .last = p;
                next[];
            };
        };

        if print_full {
            std:displayln ~ wrap_text_char std:str:trim[p.text] 70;
        } {
            if print {
                std:displayln ~ wrap_text_char std:str:trim[last.text] 70;
                std:displayln ~ wrap_text_char std:str:trim[p.text] 70;
            };
        };

        if not[p.payload] {
            .last = p;
            next[];
        };

        .last = p;

        !probs = parse_probs p.probs;
        !(word, prob) = get_word probs;

        if is_none[categories.(p.payload.category)] {
            categories.(p.payload.category) = $[0.0, 0];
        };

        if print &and cfg.m {
            std:displayln ~ std:ser:json p.min_p_tokens.res;
        };

        !expected = std:str:to_lowercase p.payload.expected;

        !judgement = if std:str:to_lowercase[word] != expected {
            categories.(p.payload.category).1 += 1;
            "FAIL";
        } {
            categories.(p.payload.category).0 += prob;
            categories.(p.payload.category).1 += 1;

            if prob < 0.8 {
                "UGLY"
            } ~ if prob < 0.9 {
                "BAD";
            } {
                "OK"
            };
        };
        if print {
            std:displayln "^^^^" ($F"{:4} expected: {}, got: {} @ {:9.7} | context={:4}" judgement p.payload.expected word prob p.prompt_token_count);
        };

        if is_some[p.top_token_results] {
            iter top_res (std:reverse p.top_token_results) {
                !tprobs = parse_probs top_res.probs;
                !(word, prob) = get_word tprobs;
                std:displayln ~ $F"       - {:9.7!f}: {}" prob word;
                if cfg.m {
                    std:displayln "          " ($F"{:6.4!f} {:3}" tprobs.0.1 tprobs.0.0);
                    iter mp (1 => -1 top_res.min_p_tokens.res) {
                        !mp_probs = parse_probs mp;
                        !mpstr = $@s range 0 3 1 {
                            if _ > 0 { $+ " | " };
                            !mpword = mp_probs.(_).0;
                            .mpword = "\n" => "\\n" mpword;
                            $+ ($F"{:6.4!f} {:3}" mp_probs.(_).1 mpword);
                        };
                        std:displayln "          " mpstr;
                    };
                };
            };
        };
    };
    categories
};

!calc_scores = {!(categories) = @;
    !cat_scores = ${};
    !keys = std:keys categories;
    std:sort { std:cmp:str:asc _ _1 } keys;
    iter cat keys {
        !p = categories.(cat).0;
        !cnt = categories.(cat).1;
        cat_scores.(cat) = ${
            score = ((100.0 * p) / cnt), category = cat, count = cnt
        };
    };
    !global_score = $@f iter vk category_weights {
        !cat = categories.(vk.1);
        if is_some[cat] {
            $+ vk.0 * cat.0 / cat.1;
        };
    };
    cat_scores._sorted_keys = keys;
    cat_scores._global = 100.0 * global_score;
    cat_scores
};

!print_categories = {!(file, categories) = @;
    std:displayln "SCORE:" file;
    !scores = calc_scores categories;

    iter key scores._sorted_keys {
        !score = scores.(key);
        std:displayln ~ $F"{:5.1!f} - {} [{}]" score.score score.category score.count;
    };

    std:displayln ~ $F"ALC-IQ4={:5.1!f}" scores._global;
};

!stat_file = {!(filepath, print, print_full) = @;
    !r = std:deser:json ~ std:io:file:read_text filepath;
    !categories = ${};

    !seeds = $@map iter r r.results { $+ str[r.seed] $t; };
    .seeds = map {|| _1 } seeds;
    std:sort { std:cmp:num:asc _ _1 } seeds;

    !test_ids = $@map iter r r.results { $+ r.test_id $t; };
    .test_ids = map {|| _1 } test_ids;
    std:sort { std:cmp:str:asc _ _1 } test_ids;

    iter res r.results {
        if is_some[cfg.test-id] {
            if cfg.test-id &> $r/$^$+[0-9]$$/ {
                if res.test_id != test_ids.(int cfg.test-id) {
                    next[];
                };
            } {
                if res.test_id != cfg.test-id {
                    next[];
                };
            };
        };

        if is_some[cfg.seed] {
            !check_seed = cfg.seed;
            if int[cfg.seed] < 99 {
                .check_seed = seeds.(int[cfg.seed]);
            };
            if str[res.seed] != check_seed {
                next[];
            };
        };

        if cfg.tree {
            std:displayln ~ $F"seed={:<10} - {}" res.seed res.test_id ;
            !categories = stat_prompts res.prompt;
            print_categories filepath categories;
        };

        stat_prompts res.prompt categories print print_full;
    };
    categories
};

if cfg._cmd == "list" {
    iter file files {
        std:displayln "FILE:" file;
        !r = std:deser:json ~ std:io:file:read_text file;
        !topics = ${};
        !categories = ${};
        !test_ids = ${};
        !seeds = ${};
        iter res r.results {
            seeds.(str[res.seed]) = $t;
            test_ids.(res.test_id) = $t;

            iter p res.prompt {
                if p.payload {
                    categories.(p.payload.category) = $t;
                    topics.(p.payload.topic) = $t;
                };
            };
        };
        .topics = map {|| _1 } topics;
        std:sort { std:cmp:str:asc _ _1 } topics;
        .categories = map {|| _1 } categories;
        std:sort { std:cmp:str:asc _ _1 } categories;
        .test_ids = map {|| _1 } test_ids;
        std:sort { std:cmp:str:asc _ _1 } test_ids;
        .seeds = map {|| int _1 } seeds;
        std:sort { std:cmp:num:asc _ _1 } seeds;

        std:displayln "Topics:";
        iter t topics \std:displayln "   " t;
        std:displayln "Categories:";
        iter t categories \std:displayln "   " t;
        std:displayln "Test IDs:";
        iter t test_ids \std:displayln "   " t;
        std:displayln "Seeds:";
        iter t seeds \std:displayln "   " t;
    };
};

if cfg._cmd == "raw_print" {
    iter file files {
        std:displayln "FILE:" file;
        !r = std:deser:json ~ std:io:file:read_text file;
        iter res r.results {
            std:displayln "### START ################";
            std:displayln res.response;
            std:displayln "### END ################";
        };
    };
    return $n;
};

if cfg._cmd == "rank" {
    !file_scores = $[];
    iter file files {
        !categories = stat_file file $f $f;
        !scores = calc_scores categories;
        scores._file = file;
        file_scores +> scores;
    };

    std:sort { std:cmp:num:desc _._global _1._global } file_scores;
    iter score file_scores {
        std:displayln ~ $F"{:5.1!f} - {}" score._global score._file;
    };

    return $n;
};

!do_print = cfg._cmd == "print" &or cfg._cmd == "print_full";

iter file files {
    std:displayln "FILE:" file;
    !categories = stat_file file do_print cfg._cmd == "print_full";
    std:displayln "--------------";
    print_categories file categories;
};

