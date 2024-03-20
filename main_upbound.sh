
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_upbound.py \
  --num_tasks 1 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --project semi-IPC-upbound \
  --epochs 50 \
  --num_gpus 4 \
  --batch_size 256 &


#python main_upbound.py \
#  --num_tasks 1 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
#  --pretrained_method byol \
#  --dataset imagenet100 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
#  --project semi-IPC-upbound \
#  --epochs 50 \
#  --num_gpus 8 \
#  --batch_size 256

#
CUDA_VISIBLE_DEVICES=4,5,6,7 python main_upbound.py \
  --num_tasks 1 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --project semi-IPC-upbound \
  --epochs 50 \
  --num_gpus 4 \
  --batch_size 256
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main_upbound.py \
#  --num_classes 200 \
#  --num_tasks 1 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/semi-IPC-v9/byol-cub100-resnet50.ckpt \
#  --pretrained_method byol \
#  --dataset cub200 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/cub200 \
#  --project semi-IPC-upbound \
#  --epochs 50 \
#  --num_gpus 4 \
#  --batch_size 256




