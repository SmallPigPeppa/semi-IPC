#
#
#for lambda in 1 3 10 20 50 100 200; do
#
#    CUDA_VISIBLE_DEVICES=7 python main_mini_v8_s5_abnum.py \
#    --num_tasks 5 \
#    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
#    --pretrained_method byol \
#    --dataset mini \
#    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
#    --pl_lambda 0.5 \
#    --project semi-IPC-abnum \
#    --epochs 50 \
#    --num_sample $lambda \
#    --perfix $lambda- \
#    --cpn_initial means &
#
#
#  CUDA_VISIBLE_DEVICES=7 python main_mini_v8_s5_abnum.py \
#    --num_tasks 10 \
#    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
#    --pretrained_method byol \
#    --dataset mini \
#    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
#    --pl_lambda 0.5 \
#    --project semi-IPC-abnum \
#    --epochs 50 \
#    --num_sample $lambda \
#    --perfix $lambda- \
#    --cpn_initial means
#done
#

for lambda in 600; do

    CUDA_VISIBLE_DEVICES=0 python main_mini_v8_s5_abnum.py \
    --num_tasks 5 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
    --pretrained_method byol \
    --dataset mini \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
    --pl_lambda 0.5 \
    --project semi-IPC-abnum \
    --epochs 50 \
    --num_sample $lambda \
    --perfix $lambda- \
    --cpn_initial means


  CUDA_VISIBLE_DEVICES=0 python main_mini_v8_s5_abnum.py \
    --num_tasks 10 \
    --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/test-code/solo-learn-latest/trained_models/byol/frxj6kgh/byol-imagenet-mini-frxj6kgh-ep=999.ckpt \
    --pretrained_method byol \
    --dataset mini \
    --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/miniImageNet \
    --pl_lambda 0.5 \
    --project semi-IPC-abnum \
    --epochs 50 \
    --num_sample $lambda \
    --perfix $lambda- \
    --cpn_initial means
done





