
CUDA_VISIBLE_DEVICES=0 python main_cifar100_origin_v8_s5_ab_nme.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 1 \
  --perfix nme- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_cifar100_origin_v8_s5_ab_nme.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 1 \
  --perfix nme- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=0 python main_cifar100_origin_v8_s5_ab_noiu.py \
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
  --perfix noiu- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=0 python main_cifar100_origin_v8_s5_ab_noiu.py \
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
  --perfix noiu- \
  --cpn_initial means




# imagenet100
CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8_ab_nme.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix nme- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8_ab_nme.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix nme- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8_ab_noiu.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix noiu- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8_ab_noiu.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix noiu- \
  --cpn_initial means


# miniImageNet

CUDA_VISIBLE_DEVICES=0 python main_mini_v8_s5_ab_nme.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 1 \
  --perfix nme- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_mini_v8_s5_ab_nme.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 1 \
  --perfix nme- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_mini_v8_s5_ab_noiu.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix noiu- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=0 python main_mini_v8_s5_ab_noiu.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0. \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix noiu- \
  --cpn_initial means