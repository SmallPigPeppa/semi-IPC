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

#CUDA_VISIBLE_DEVICES=0 python main_imagenet100_v8_ood.py \
#  --num_tasks 5 \
#  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
#  --pretrained_method byol \
#  --dataset imagenet100 \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
#  --pl_lambda 0.2 \
#  --project semi-IPC-ood \
#  --epochs 50 \
#  --perfix v8-semi-dual- \
#  --cpn_initial means \
#  --num_ood 65 \
#  --data_path_ood /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet-subset-new


#for n in 65 130 260  520; do
CUDA_VISIBLE_DEVICES=1 python main_imagenet100_v8_ood.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --project semi-IPC-ood \
  --epochs 50 \
  --perfix v8-semi-dual- \
  --cpn_initial means \
  --num_ood 65 \
  --data_path_ood /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet-subset-new &


CUDA_VISIBLE_DEVICES=2 python main_imagenet100_v8_ood.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
  --pretrained_method byol \
  --dataset imagenet100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
  --pl_lambda 0.2 \
  --project semi-IPC-ood \
  --epochs 50 \
  --perfix v8-semi-dual- \
  --cpn_initial means \
  --num_ood 130 \
  --data_path_ood /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet-subset-new &

CUDA_VISIBLE_DEVICES=3 python main_imagenet100_v8_ood.py \
--num_tasks 5 \
--pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
--pretrained_method byol \
--dataset imagenet100 \
--data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
--pl_lambda 0.2 \
--project semi-IPC-ood \
--epochs 50 \
--perfix v8-semi-dual- \
--cpn_initial means \
--num_ood 260 \
--data_path_ood /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet-subset-new &


CUDA_VISIBLE_DEVICES=4 python main_imagenet100_v8_ood.py \
--num_tasks 5 \
--pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/3tx0at58/byol-imagenet-3tx0at58-ep=999.ckpt \
--pretrained_method byol \
--dataset imagenet100 \
--data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet100 \
--pl_lambda 0.2 \
--project semi-IPC-ood \
--epochs 50 \
--perfix v8-semi-dual- \
--cpn_initial means \
--num_ood 520 \
--data_path_ood /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet-subset-new