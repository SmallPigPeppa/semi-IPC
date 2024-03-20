
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_upboud.py \
  --num_tasks 1 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --dataset cifar100 \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --project semi-IPC-upbound \
  --epochs 50 \
  --num_gpus 4 \
  --batch_size 256








