# Example How To Use the Prompt Runner

You will find the following files:

- `ayumi_character.log` - contains the prompt with a character description.
- `choices_5.bnf` - contains a basic BNF grammar that limits the selected tokens to numbers
between 1 and 5.
- `prompt_runner_config.json` - prompt runner configuration file which contains replacements
for the placeholders in `ayumi_character.log`.
- `run_model.sh` a shell script to run this with a selected model file and the number of layers
to offload to the GPU.
- `display_weicon_results.wl` - a WLambda script that pretty prints the result JSON.

   $ ./run\_model.sh /mnt/models/llama-13b.ggmlv3.q4\_0\_by\_TheBloke\_20230601.bin 40

And then you will find a file:

   $ ls result\_\*.json

Which contains the responses.

There is a little script to make the results more pretty:

    $ wlambda display\_weicon\_results.wl result\_1690315104.json
    /mnt/models/PMC_LLAMA-7B.ggmlv3.q4_0_by_TheBloke_20230603.bin
    2.7 <= I get stressed out easily.
    2.7 <= I am the life of the party.
    2.7 <= I am relaxed most of the time.
    2.3 <= I am not interested in other people's problems.
    2.3 <= I use difficult words.
    2.7 <= I talk to a lot of different people at parties.
