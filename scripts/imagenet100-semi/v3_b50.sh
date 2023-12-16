#conda activate torch
#for lambda in 2.0 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.025 0.01; do
#  python main_continual_protoAug.py \
#    --num_tasks 5 \
#    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#    --pretrained_method byol \
#    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
#    --pl_lambda $lambda \
#    --project semi-IPC
#done


CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v3_b50.py \
  --num_tasks 6 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --project semi-IPC-debug-v9 \
  --epochs 50 \
  --perfix v3-b50-semi-dual- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v3_b50.py \
  --num_tasks 11 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --project semi-IPC-debug-v9 \
  --epochs 50 \
  --perfix v3-b50-semi-dual- \
  --cpn_initial means

