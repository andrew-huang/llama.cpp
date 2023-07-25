!data = std:deser:json ~ std:io:file:read_text @@.0;
#std:displayln data;

!ids = $[];
iter cfg data.config.prompt_tests {
    std:push ids cfg.id => cfg.replacements.0.1;
};

!response_by_id = ${};
iter r data.results {
#    std:displayln r;
    if is_none[response_by_id.(r.test_id)] {
        response_by_id.(r.test_id) = $[];
    };
    std:push response_by_id.(r.test_id) r.response;
};

!avg_lst = {
    !sum = $@i iter n _ { $+ int[n]; };
    float[sum] / len[_]
};

std:displayln data.model_file;

iter question ids {
    !cur = avg_lst[response_by_id.(question.0)];
    std:displayln ~ $F"{:3.1!f} <= {}" cur question.1;
};
