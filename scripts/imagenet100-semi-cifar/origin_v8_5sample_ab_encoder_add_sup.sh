

CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/semi-IPC-v9/sup/checkpoints/resnet50-cifar/test.ckpt \
  --pretrained_method sup \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means &


CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_ab.py \
  --num_tasks 10 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/semi-IPC-v9/sup/checkpoints/resnet50-cifar/test.ckpt \
  --pretrained_method sup \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.02 \
  --project semi-IPC-ab-encoder \
  --epochs 50 \
  --perfix s5-v8- \
  --cpn_initial means



