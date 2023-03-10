#conda activate torch


python main_continual.py \
      --num_tasks 5 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda 0 \
      --project IPC-ablation-exp \
      --epochs 300 \
      --cpn_initial means \
      --perfix ipc

python main_continual.py \
      --num_tasks 10 \
      --pretrained_model /share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt \
      --pretrained_method byol\
      --pl_lambda 0 \
      --project IPC-ablation-exp \
      --epochs 300 \
      --cpn_initial means \
      --perfix ipc