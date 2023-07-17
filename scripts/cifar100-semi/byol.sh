conda activate torch
python main_continual_semi.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol \
      --data_path /share/wenzhuoliu/torch_ds \
      --project semi-IPC

#python main_continual_semi.py \
#      --num_tasks 10 \
#      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
#      --pretrained_method byol
