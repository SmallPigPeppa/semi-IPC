import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils.dataset_utils_v2_cifar import get_dataset, get_pretrained_dataset, split_dataset, get_dual_dataset, \
    get_dataset_std
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.encoder_utils import get_pretrained_encoder
from utils.args_utils import parse_args_cpn
from models.icpn_protoAug_semi_2view_blance_add_dualloss_v8_upbound import mFC


def main():
    seed_everything(5)
    args = parse_args_cpn()
    if "cifar" in args.dataset:
        encoder = get_pretrained_encoder(args.pretrained_model, cifar=True)
    else:
        encoder = get_pretrained_encoder(args.pretrained_model, cifar=False)

    model = mFC(**args.__dict__)

    # classes_order = torch.tensor(
    #     [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28,
    #      53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97,
    #      2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7,
    #      63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])
    # classes_order = torch.randperm(num_classes)
    # classes_order = torch.tensor(list(range(args.num_classes)))
    # tasks_initial = classes_order[:int(args.num_classes / 2)].chunk(1)
    # tasks_incremental = classes_order[int(args.num_classes / 2):args.num_classes].chunk(args.num_tasks)
    # tasks = tasks_initial + tasks_incremental
    # tasks = classes_order.chunk(args.num_tasks)
    train_dataset, test_dataset = get_dataset_std(dataset=args.dataset, data_path=args.data_path)
    task_idx = 0

    # for task_idx in range(0, args.num_tasks):
    #     model.tasks = tasks
    #     model.current_task_idx = task_idx
    wandb_logger = WandbLogger(
        name=f"{args.perfix}{args.dataset}-{args.pretrained_method}-{args.num_tasks}tasks-steps{task_idx}",
        project=args.project,
        entity=args.entity,
        offline=False,
    )
    #     if args == 0:
    wandb_logger.log_hyperparams(args)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    #     train_dataset_task = split_dataset(
    #         train_dataset,
    #         tasks=tasks,
    #         task_idx=[task_idx],
    #     )
    #     test_dataset_task = split_dataset(
    #         test_dataset,
    #         tasks=tasks,
    #         task_idx=list(range(task_idx + 1)),
    #     )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        # gradient_clip_val=1.0,
        sync_batchnorm=True,
        accelerator='ddp',
        logger=wandb_logger,
        checkpoint_callback=False,
        precision=16,
        callbacks=[lr_monitor]

    )
    model.encoder = encoder
    trainer.fit(model, train_loader, test_loader)
    wandb.finish()


if __name__ == '__main__':
    main()
