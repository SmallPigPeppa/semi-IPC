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



python main_continual_protoAug_semi_2view_equal_split_real_remove_fixfeature_debugbs_blance.py \
  --num_tasks 5 \
  --pretrained_model /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
  --pretrained_method byol \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
  --pl_lambda 0.2 \
  --project semi-IPC \
  --epochs 50 \
  --perfix semi-base-equal-split-real \
  --cpn_initial means

#--pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#python main_continual_protoAug.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol
