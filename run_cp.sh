./prompt_runner -ngl 42 -f $2 --threads 10 -c 8200 -n 250 -b 512 --top-k 0 --top-p 1 --tfs 1 --min-p 0.05 --temp 1.1 --repeat-last-n 512 --repeat-penalty 1.05 -m $1 --in-prefix "$3"
