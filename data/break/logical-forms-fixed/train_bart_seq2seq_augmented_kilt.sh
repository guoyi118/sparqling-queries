#/bin/bash
python train_bart_seq2seq_augmented_kilt.py \
    --gpus 4 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 4 \
    --max_steps 200000 \
    --divergences both \
    # 2>&1 | tee models/bart_seq2seq_augmented_structured_zeroshot/log.txt
