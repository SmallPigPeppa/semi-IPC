CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
  --pretrained_method barlow \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/mocov2plus/1kguyx5e/mocov2plus-imagenet32-1kguyx5e-ep=999.ckpt \
  --pretrained_method moco \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
  --pretrained_method simclr \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=1000.ckpt \
  --pretrained_method simsiam \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
  --pretrained_method swav \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt \
  --pretrained_method barlow \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means

CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
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


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt \
  --pretrained_method simclr \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=1000.ckpt \
  --pretrained_method simsiam \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
   --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt \
  --pretrained_method swav \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means
