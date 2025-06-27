python3 train_st.py   \
    --lr 8e-5   \
    --model_name /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-roberta-output-100k-finetuned-alpha15/checkpoint-80000   \
    --alpha 1.5 \
    --pre_iter 5   \
    --post_iter 5   \
    --output_dir /mnt/scratch-artemis/mtreviso/sparse_pretraining/st-roberta-output-100k-finetuned-alpha15   \
    --fp16


python3 train_st.py   \
    --lr 8e-5   \
    --model_name /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-roberta-output-100k-finetuned/checkpoint-80000   \
    --alpha 2.0 \
    --pre_iter 5   \
    --post_iter 5   \
    --output_dir /mnt/scratch-artemis/mtreviso/sparse_pretraining/st-roberta-output-100k-finetuned-alpha20   \
    --fp16



