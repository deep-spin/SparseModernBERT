python3 train_pylate.py \
  --lr 8e-5 \
  --num_train_epochs 1 \
  --batch_size 16 \
  --accum_steps 1 \
  --model_name /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-modernbert-output-100k-finetuned \
  --alpha 2.0 \
  --use_triton_entmax \
  --pre_iter 5 \
  --post_iter 5 \
  --output_dir /mnt/scratch-artemis/mtreviso/sparse_pretraining/pylate-modernbert-100k-finetuned-alpha20


python3 train_pylate.py \
  --lr 8e-5 \
  --num_train_epochs 1 \
  --batch_size 16 \
  --accum_steps 1 \
  --model_name /mnt/scratch-artemis/mtreviso/sparse_pretraining/mlm-modernbert-output-100k-finetuned-alpha15 \
  --alpha 1.5 \
  --use_triton_entmax \
  --pre_iter 5 \
  --post_iter 5 \
  --output_dir /mnt/scratch-artemis/mtreviso/sparse_pretraining/pylate-modernbert-100k-finetuned-alpha15
