CUDA_VISIBLE_DEVICES=5 python main_imagenet100_v8_ood.py \
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
--num_ood 750 \
--data_path_ood /mnt/mmtech01/usr/liuwenzhuo/torch_ds/imagenet-subset-new