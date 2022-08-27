for lambda in 0.2 0.1 0.05 0.025 0.01 0.005 0.0025 0.001
do
    data_path=/share/wenzhuoliu/torch_ds
    pretrained_dir=/share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572
    pretrained_path="$(ls $pretrained_dir/*.ckpt)"
    echo "pretrained_path: $pretrained_path"
    /share/wenzhuoliu/conda-envs/solo-learn/bin/python main_continual.py \
        --num_tasks 6\
        --meathod cpn \
        --dataset cifar100 \
        --encoder resnet50 \
        --train_data_path ${data_path} \
        --val_data_path ${data_path} \
        --max_epochs 100 \
        --devices 0 \
        --accelerator gpu \
        --precision 16 \
        --optimizer sgd \
        --scheduler step \
        --lr 0.5 \
        --lr_decay_steps 60 80 \
        --weight_decay 0 \
        --batch_size 256 \
        --num_workers 10 \
        --crop_size 32 \
        --name cpn-cifar-lambda${lambda} \
        --pretrained_feature_extractor ${pretrained_path} \
        --project Incremental-CPN \
        --entity pigpeppa \
        --wandb \
        --mean 0.5071 0.4867 0.4408 \
        --std 0.2675 0.2565 0.2761 \
        --pl_lambda ${lambda}
done