

# +pl + dual + pa
CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

# +pl + dual
CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --pa_lambda 0. \
  --dual_lambda 1.0 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --pa_lambda 0. \
  --dual_lambda 1.0 \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means






# +pl
CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path  /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --pa_lambda 0. \
  --dual_lambda 0. \
  --project semi-IPC-ab \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means

# +None
CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
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
  --perfix v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ab.py \
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
  --perfix v8- \
  --cpn_initial means