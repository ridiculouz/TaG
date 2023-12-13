python src/train_tag.py \
    --model_name_or_path roberta-base \
    --num_epoch 50 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.04 \
    --save_path ./model/roberta-base-tag.pt \
    --notes roberta-base