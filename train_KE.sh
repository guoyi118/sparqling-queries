#/bin/bash
python -m  text2qdmr.commands.knowledgeeditor \
    --gpus 1 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 64 \
    --max_steps 50000 \
    --divergences both \
    --use_views \
    2>&1 | output.txt

CORENLP_HOME=text2qdmr/third_party/stanford-corenlp-full-2018-10-05 python -m  text2qdmr.commands.knowledgeeditor --gpus 1 --max_steps 10000 > output.txt