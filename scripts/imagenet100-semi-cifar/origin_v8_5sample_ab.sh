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


## +pl + dual + pa
#CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
#  --num_tasks 10 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#  --pretrained_method byol \
#  --dataset cifar100 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
#  --pl_lambda 0.02 \
#  --project semi-IPC-ab \
#  --epochs 50 \
#  --perfix s5-v8- \
#  --cpn_initial means
#
#CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
#  --num_tasks 5 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#  --pretrained_method byol \
#  --dataset cifar100 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
#  --pl_lambda 0.02 \
#  --project semi-IPC-ab \
#  --epochs 50 \
#  --perfix s5-v8- \
#  --cpn_initial means

# +pl + dual
CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --pa_lambda 0. \
  --dual_lambda 1.0 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --pa_lambda 0. \
  --dual_lambda 1.0 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means






# +pl
CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means

# +None
CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=2 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means