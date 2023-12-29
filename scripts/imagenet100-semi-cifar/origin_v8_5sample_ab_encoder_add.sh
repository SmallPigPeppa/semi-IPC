

CUDA_VISIBLE_DEVICES=0 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
  --pretrained_method moco \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means &


CUDA_VISIBLE_DEVICES=1 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
  --pretrained_method moco \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


