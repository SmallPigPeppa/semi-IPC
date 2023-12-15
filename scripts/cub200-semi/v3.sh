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
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
CUDA_VISIBLE_DEVICES=0 python main_cub200_v3.py \
  --num_classes 200 \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/semi-IPC-v9/cub200-resnet50.pth \
  --pretrained_method byol \
  --dataset cub200 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/cub200 \
  --pl_lambda 0.2 \
  --project semi-IPC-debug-v9 \
  --epochs 50 \
  --perfix v3-semi-dual- \
  --cpn_initial means


#CUDA_VISIBLE_DEVICES=0 python main_cub200_v3.py \
#  --num_classes 200 \
#  --num_tasks 1 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/semi-IPC-v9/cub200-resnet50.pth \
#  --pretrained_method byol \
#  --dataset cub200 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/cub200 \
#  --pl_lambda 0.2 \
#  --project semi-IPC-debug-v9 \
#  --epochs 1 \
#  --perfix v3-semi-dual- \
#  --cpn_initial means


