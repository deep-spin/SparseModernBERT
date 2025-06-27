python3 train_st.py \
  --lr 8e-5 \
  --model_name /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-modernbert-output-100k-finetuned \
  --alpha 2.0 \
  --use_triton_entmax \
  --pre_iter 5 \
  --post_iter 5 \
  --output_dir /mnt/scratch-artemis/mtreviso/sparse_pretraining/st-modernbert-100k-finetuned-alpha20


python3 train_st.py \
  --lr 8e-5 \
  --model_name /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-modernbert-output-100k-finetuned-alpha15 \
  --alpha 1.5 \
  --use_triton_entmax \
  --pre_iter 5 \
  --post_iter 5 \
  --output_dir /mnt/scratch-artemis/mtreviso/sparse_pretraining/st-modernbert-100k-finetuned-alpha15
