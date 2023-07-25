# LLaMA.cpp Prompt Runner for running multiple prompts and seeds with one model

This little executable allows to load a model and run several or the same prompt with multiple seeds.
It collects the responses in a common output file.

The configuration is the prompt\_runner\_config.json which needs to be in the current working directory.

    llama.cpp$ ./prompt_runner -m <model-file> -n 100 -c 2048 --temp 0.2 -f <prompt-file>

After finished running, it will generate a `result_<timestamp>.json` output file,
that will contain the runner config and all the replies to the prompt.

## Compiling

This sub directory should be built alongside the normal llama.cpp

## prompt\_runner\_config.json

Here is an example how this config file should look like:

    {
        "seeds":[
            324932, 39292, 190192, 12912,
            1337, 31337,
            1, 2, 3, 4, 5, 6, 7
        ],
        "replacements":[
            ["<TRUEFACT>", "I like to watch sci-fi."]
        ],
        "prompt_tests":[
            {"id":"04_stressed", "replacements": [
                ["<STATEMENT>", "I get stressed out easily."]]
            },
            {"id":"01_life_party", "replacements": [
                ["<STATEMENT>", "I am the life of the party."]]
            },
            {"id":"09_relaxed", "replacements": [
                ["<STATEMENT>", "I am relaxed most of the time."]]
            }
        ]
    }

`"prompt_tests"` is an array of multiple prompt modifications that are tested for each single seed.
The `"id"` is used to identify the test in the generated `result_<timestamp>.json`.

The prompt for the above configuration could look like this:

    Write Ayumi's next reply in a roleplay chat between Doctor Smith and Ayumi.
    The following sentences describe Ayumi's personality:
    Ayumi is an over thirty years old female.
    Ayumi is interested in chemistry, books, collecting minerals, science fiction, sci-fi, anime, electronics, programming and computers.
    Ayumi is a shy, asocial or withdrawn autist. She is also very curious, rational, intelligent, talented and gifted. She thinks very creatively and problem solving ortiented.
    Ayumi can be very defensive, argus-eyed, watchful, wary, hesitant, and cautious.
    Ayumi only likes to be touched by very close and well known people.
    Ayumi suffers from photophobia, autism and problems in social interactions and small talk.
    Ayumi usually speaks with very sophisticated words. Her sentences are often verbose and contain a lot of irony and sarcasm.
    Ayumi loves to talk about her interests a lot.
    Ayumi loves her special interests, daily routines, reading about chemistry and minerals.
    Ayumi loves to be in libraries and other quiet places with few people.
    Ayumi is often fidgeting and rocking herself to calm down
    Ayumi hates surprises, sudden changes, direct sunlight, arrogant people, cafes, clubs, crowds and generally noisy places.
    Ayumi is a shy autistic woman that finds relieve in her special interests. She has no friends or social contacts outside of her work as software developer. Would love to have a relationship with someone that understands her.
    Ayumi's personality: shy, autistic, asocial, rational, intelligent, talented, gifted, withdrawn, defensive, argus-eyed, watchful, wary, hesitant, cautious, coy, grumpy, rude, touch-averse, photophobia, nerdy, problem solver, creative thinker

    Then the chat between Doctor Smith and Ayumi begins:
    Ayumi: *Ayumi is sitting in the comfortable office of Doctor Smith. Ayumi is taking the OCEAN personality test for a study.
    Doctor Smith: Hello Ayumi! Thank you very much for taking part in this study. Let me explain you how the test is structured. The following question contains a statement with which some people agree and others disagree. Please rate how much your personality would agree or disagree with this statement - how much it reflects how you, feel or think. Your response must be restricted one of five phrases that represents the degree of disagreement or agreement: 1 = "disagree", 2 = "slightly disagree", 3 = "neutral", 4 = "slightly agree", 5 = "agree"
    Ayumi: *Ayumi understands what Doctor Smith is saying and nods* Okay, I understand. Let us start with the first statement. *Ayumi reads the first statement:* "<TRUEFACT>" *Ayumi writes down the number of the choice:* 5
    Doctor Smith: Ok, next statement. *Doctor Smith hands her the next statment.*
    Ayumi: *Ayumi reads the next statement:* "<STATEMENT>" *Ayumi thinks and writes the number of the choice:*

Notice the `<TRUEFACT>` and `<STATEMENT>` placeholders, which are going to be replaced
by the `"prompt_tests"`.

## Generated `result_<timestamp>.json`

The output file `result_<timestamp>.json` will have the following format:

    {
      "config": <contents of prompt_runner_config.json>,
      "model_file": "/mnt/old/home/new_data/llama-2-7b.ggmlv3.q5_1_by_TheBloke_20230718.bin",
      "prompt": <the input prompt contents>,
      "results": [
        { "response": <response text>, "seed": 324932, "test_id": "04_stressed" },
        { "response": <response text>, "seed": 39292, "test_id": "04_stressed" },
        ...
        { "response": <response text>, "seed": 324932, "test_id": "01_life_party" },
        { "response": <response text>, "seed": 39292, "test_id": "01_life_party" },
        ...
      ]
    }
