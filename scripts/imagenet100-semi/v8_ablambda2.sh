
  CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8.py \
    --num_tasks 10 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
    --pretrained_method byol \
    --dataset imagenet100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
    --pl_lambda 2.0 \
    --project semi-IPC-ablambda \
    --epochs 50 \
    --perfix v8- \
    --cpn_initial means


for lambda in  1.5 2.0; do

  CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8.py \
    --num_tasks 10 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
    --pretrained_method byol \
    --dataset imagenet100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
    --pl_lambda $lambda \
    --project semi-IPC-ablambda \
    --epochs 50 \
    --perfix v8- \
    --cpn_initial means

done



