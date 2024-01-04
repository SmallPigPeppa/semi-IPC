#for lambda in 1 3 10 20 50 100 200; do
#
#    CUDA_VISIBLE_DEVICES=6 python main_cifar100_origin_v8_s5_abnum.py \
#    --num_tasks 10 \
#    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#    --pretrained_method byol \
#    --dataset cifar100 \
#    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
#    --pl_lambda 0.02 \
#    --project semi-IPC-abnum \
#    --epochs 50 \
#    --num_sample $lambda \
#    --perfix $lambda- \
#    --cpn_initial means &
#
#  CUDA_VISIBLE_DEVICES=6 python main_cifar100_origin_v8_s5_abnum.py \
#    --num_tasks 5 \
#    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#    --pretrained_method byol \
#    --dataset cifar100 \
#    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
#    --pl_lambda 0.02 \
#    --project semi-IPC-abnum \
#    --epochs 50 \
#    --num_sample $lambda \
#    --perfix $lambda- \
#    --cpn_initial means
#
#done



for lambda in 500; do

    CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_abnum.py \
    --num_tasks 10 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
    --pretrained_method byol \
    --dataset cifar100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
    --pl_lambda 0.02 \
    --project semi-IPC-abnum \
    --epochs 50 \
    --num_sample $lambda \
    --perfix $lambda- \
    --cpn_initial means

  CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5_abnum.py \
    --num_tasks 5 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
    --pretrained_method byol \
    --dataset cifar100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
    --pl_lambda 0.02 \
    --project semi-IPC-abnum \
    --epochs 50 \
    --num_sample $lambda \
    --perfix $lambda- \
    --cpn_initial means

done








