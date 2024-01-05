!std_opts = $[
    $["-f", :FILE, "The input JSON file, if none provided, the lastest named 'result_*.json' in the current dir is taken."],
    $["-m", "--min-p", "Prints the min-p tokens"],
    $["-t", :TOP_N, $o(1)],
];
!cfg = std:app:simple_cli
    "print_qa" "1.0.0"
    "Printing a prompt_runner/query_runner response file's IQ question results."
    :stat => $[*std_opts]
    :raw_print => $[*std_opts]
    :print => $[*std_opts];

!files = if std:str:len[cfg.f] > 0 {
    $[cfg.f]
} {
    !files = $[];
    std:fs:read_dir "." {
        if _.name &> $r/$^result_*\.json$$/ {
            files +> _.mtime => _.path;
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

!stat_file = {!(filepath, print) = @;
    !r = std:deser:json ~ std:io:file:read_text filepath;
    !categories = ${};
    iter res r.results {
        iter p res.prompt {
            if print {
                std:displayln ~ wrap_text_char std:str:trim[p.text] 70;
            };
            if p.payload {
                !probs = parse_probs p.probs;
                !(word, prob) = get_word probs;

                if is_none[categories.(p.payload.category)] {
                    categories.(p.payload.category) = $[0.0, 0];
                };

                if print &and cfg.m {
                    std:displayln ~ std:ser:json p.min_p_tokens.res;
                };

                .word = std:str:to_lowercase word;
                !expected = std:str:to_lowercase p.payload.expected;

                !judgement = if word != expected {
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
            };
        };
    };
    categories
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

!do_print = cfg._cmd == "print";

iter file files {
    std:displayln "SCORE:" file;
    !categories = stat_file file do_print;
    !keys = std:keys categories;
    std:sort { std:cmp:str:asc _ _1 } keys;
    iter cat keys {
        !p = categories.(cat).0;
        !cnt = categories.(cat).1;
        std:displayln ~ $F"{:5.1!f} - {}" ((100.0 * p) / cnt) cat;
    }
};
