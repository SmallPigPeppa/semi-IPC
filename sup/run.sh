python resnet50.py \
  --num_gpus 8 \
  --weight_decay 2e-5 \
  --project semi-ipc-super \
  --num_workers 8 \
  --batch_size 128 \
  --checkpoint_dir checkpoints/resnet50-cifar \
  --run_name resnet50-cifar \
  --max_epochs 90 \
  --learning_rate 0.5
