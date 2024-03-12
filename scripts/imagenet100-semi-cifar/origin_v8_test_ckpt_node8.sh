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



#CUDA_VISIBLE_DEVICES=0 python main_cifar100_origin_v8_test.py \
#  --num_tasks 5 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#  --pretrained_method byol \
#  --dataset cifar100 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
#  --pl_lambda 0.02 \
#  --project null \
#  --epochs 50 \
#  --perfix v8- \
#  --cpn_initial means



CUDA_VISIBLE_DEVICES=1 python main_cifar100_origin_v8_test_ckpt.py \
  --num_tasks 5 \
  --pretrained_model /lustre/home/wzliu/test-project/ssl-pretrain/cifar100/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /lustre/home/wzliu/torch_ds \
  --pl_lambda 0.02 \
  --project test-cifar100 \
  --epochs 50 \
  --perfix v8- \
  --cpn_initial means






