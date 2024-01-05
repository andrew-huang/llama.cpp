!cfg = std:app:simple_cli
    "print_qa" "1.0.0"
    "Printing a prompt_runner/query_runner response file's IQ question results."
    :print => $[
        $["-f", :FILE, "The input JSON file, if none provided, the lastest named 'result_*.json' in the current dir is taken."]
    ];

!filepath = if cfg.f {
    cfg.f
} {
    !files = $[];
    std:fs:read_dir "." {
        if _.name &> $r/$^result_*\.json$$/ {
            std:displayln _.mtime;
            files +> _.mtime => _.path;
        };
        $f
    };

    std:sort { std:cmp:num:desc _.0 _1.0 } files;
    files.0.1
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

std:displayln filepath;
!r = std:deser:json ~ std:io:file:read_text filepath;
iter res r.results {
    iter p res.prompt {
        std:displayln std:str:trim[p.text];
        if p.payload {
            !probs = parse_probs p.probs;
            std:displayln "   ###" p.payload.expected probs.0 "   " p.prompt_token_count;
            if probs.0.0 != p.payload.expected {
                std:displayln "   *** FAIL";
            } {
                if probs.0.1 < 0.9 {
                    std:displayln "   *** UGLY";
                } ~ if probs.0.1 < 0.95 {
                    std:displayln "   *** BAD";
                } ~ if probs.0.1 < 0.8 {
                    std:displayln "   *** VERY BAD";
                };
            };
        };
    };
};
