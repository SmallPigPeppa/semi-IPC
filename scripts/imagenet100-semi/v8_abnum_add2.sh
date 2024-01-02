
for lambda in 100 200; do
  CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8_abnum.py \
    --num_tasks 5 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
    --pretrained_method byol \
    --dataset imagenet100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
    --pl_lambda 0.2 \
    --project semi-IPC-abnum \
    --epochs 50 \
    --num_sample $lambda \
    --perfix $lambda- \
    --cpn_initial means

done



