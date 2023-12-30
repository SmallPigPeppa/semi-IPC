for lambda in 0.01 0.05 0.1 0.5 1.0 2.0; do

    CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5.py \
    --num_tasks 10 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
    --pretrained_method byol \
    --dataset cifar100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
    --pl_lambda $lambda \
    --project semi-IPC-ablambda \
    --epochs 50 \
    --perfix s5-v8- \
    --cpn_initial means &

  CUDA_VISIBLE_DEVICES=3 python main_cifar100_origin_v8_s5.py \
    --num_tasks 5 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
    --pretrained_method byol \
    --dataset cifar100 \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
    --pl_lambda $lambda \
    --project semi-IPC-ablambda \
    --epochs 50 \
    --perfix s5-v8- \
    --cpn_initial means

done








