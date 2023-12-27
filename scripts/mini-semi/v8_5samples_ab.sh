


# +pl + dual + pa
CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0.5\
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0.5\
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

# +pl + dual
CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0.5\
  --pa_lambda 0. \
  --dual_lambda 1.0 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0.5\
  --pa_lambda 0. \
  --dual_lambda 1.0 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means






# +pl
CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0.5\
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
  --pretrained_method byol \
  --dataset mini \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
  --pl_lambda 0.5\
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

# +None
CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
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
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_mini_v8_s5_ab.py \
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
  --perfix v8- \
  --cpn_initial means