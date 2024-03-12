

#for lambda in 0.01 0.05 0.1 0.5 1.0 2.0; do
for lambda in  0.2 0.8 1.5; do

    CUDA_VISIBLE_DEVICES=2 python main_mini_v8_s5.py \
    --num_tasks 5 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
    --pretrained_method byol \
    --dataset mini \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
    --pl_lambda $lambda \
    --project semi-IPC-ablambda \
    --epochs 50 \
    --perfix s5-v8- \
    --cpn_initial means &


  CUDA_VISIBLE_DEVICES=2 python main_mini_v8_s5.py \
    --num_tasks 10 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
    --pretrained_method byol \
    --dataset mini \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
    --pl_lambda $lambda \
    --project semi-IPC-ablambda \
    --epochs 50 \
    --perfix s5-v8- \
    --cpn_initial means
done



