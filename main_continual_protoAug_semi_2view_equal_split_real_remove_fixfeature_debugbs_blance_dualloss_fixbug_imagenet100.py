import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils.dataset_utils import get_dataset, get_pretrained_dataset, split_dataset, get_dual_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.encoder_utils import get_pretrained_encoder
from utils.args_utils import parse_args_cpn
from models.icpn_protoAug_semi_2view_blance_add_dualloss import IncrementalCPN
from collections import defaultdict
import random
from torch.utils.data import Subset




def keep_n_samples_per_class(dataset, n, return_means=False):
    # 首先，提取所有的标签
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]

    # 转换成tensor以加快处理速度
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # 为每个类别收集样本索引
    class_samples = defaultdict(list)
    for label in torch.unique(labels):
        label_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_samples[label.item()] = label_indices.tolist()

    new_indices = []

    # 为每个类别保留n个样本
    for label, samples in class_samples.items():
        if len(samples) <= n:
            new_indices.extend(samples)
        else:
            new_indices.extend(random.sample(samples, n))

    # 创建一个新的数据集
    new_dataset = Subset(dataset, new_indices)

    # 如果需要返回每类的均值
    if return_means:
        class_means = {}
        for label in class_samples.keys():
            selected_samples = [dataset[i][0] for i in new_indices if dataset[i][1] == label]
            class_means[str(torch.tensor(label))] = torch.mean(torch.stack(selected_samples), dim=0)

        return new_dataset, class_means
    else:
        return new_dataset


# def keep_n_samples_per_class(dataset, n, return_means=False):
#     class_samples = defaultdict(list)
#
#     # Collect samples for each class
#     for i, (sample, label) in enumerate(dataset):
#         # label = label.item()
#         if torch.is_tensor(label):
#             label = label.item()
#         class_samples[label].append(i)
#
#     new_indices = []
#
#     # Keep n samples per class
#     for label, samples in class_samples.items():
#         if len(samples) <= n:
#             new_indices.extend(samples)
#         else:
#             new_indices.extend(random.sample(samples, n))
#
#     # Create a new dataset with selected samples
#     new_dataset = Subset(dataset, new_indices)
#
#     # If return_means is True, calculate means of selected samples for each class
#     class_means = {}
#     if return_means:
#         for label in class_samples.keys():
#             selected_samples = [dataset[i][0] for i in new_indices if dataset[i][1] == label]
#             class_means[str(torch.tensor(label))] = torch.mean(torch.stack(selected_samples), dim=0)
#
#     if return_means:
#         return new_dataset, class_means
#     else:
#         return new_dataset


def main():
    seed_everything(5)
    args = parse_args_cpn()
    num_gpus = [0]
    if "cifar" in args.dataset:
        encoder = get_pretrained_encoder(args.pretrained_model, cifar=True)
    else:
        encoder = get_pretrained_encoder(args.pretrained_model, cifar=False)

    model = IncrementalCPN(**args.__dict__)

    classes_order = torch.tensor(
        [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28,
         53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97,
         2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7,
         63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])
    # classes_order = torch.randperm(num_classes)
    # classes_order = torch.tensor(list(range(args.num_classes)))
    # tasks_initial = classes_order[:int(args.num_classes / 2)].chunk(1)
    # tasks_incremental = classes_order[int(args.num_classes / 2):args.num_classes].chunk(args.num_tasks)
    # tasks = tasks_initial + tasks_incremental
    tasks = classes_order.chunk(args.num_tasks)
    train_dataset, test_dataset = get_dataset(dataset=args.dataset, data_path=args.data_path)
    dual_dataset = get_dual_dataset(dataset=args.dataset, data_path=args.data_path)

    for task_idx in range(0, args.num_tasks):
        model.tasks = tasks
        model.current_task_idx = task_idx
        model.batch_size = 64
        model.semi_batch_size = 64
        wandb_logger = WandbLogger(
            name=f"{args.perfix}{args.dataset}-{args.pretrained_method}-lambda{args.pl_lambda}-{args.num_tasks}tasks-steps{task_idx}",
            project=args.project,
            entity=args.entity,
            offline=False,
        )
        if args == 0:
            wandb_logger.log_hyperparams(args)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        print("split_dataset...")
        train_dataset_task = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=[task_idx],
        )
        test_dataset_task = split_dataset(
            test_dataset,
            tasks=tasks,
            task_idx=list(range(task_idx+1)),
        )
        dual_dataset_task = split_dataset(
            dual_dataset,
            tasks=tasks,
            task_idx=[task_idx],
        )
        print("finished...")

        train_dataset_task_fix, test_dataset_task_fix, cpn_means = get_pretrained_dataset(
            encoder=encoder,
            train_dataset=train_dataset_task,
            test_dataset=test_dataset_task,
            return_means=True)

        # train_loader = DataLoader(train_dataset_task, batch_size=64, shuffle=True)
        # test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=True)

        train_loader = DataLoader(train_dataset_task, batch_size=256, shuffle=True,pin_memory=False)
        dual_loader = DataLoader(dual_dataset_task, batch_size=256, shuffle=True,pin_memory=True,num_workers=16)
        test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=True,pin_memory=True,num_workers=8)

        print("keep_n_samples_per_class...")
        _, cpn_means = keep_n_samples_per_class(train_dataset_task_fix, n=10, return_means=True)
        supervised_data = keep_n_samples_per_class(train_dataset_task, n=10, return_means=False)
        print("finished...")
        supervised_loader = DataLoader(supervised_data, batch_size=64, shuffle=True,pin_memory=True,num_workers=8)

        train_loaders = {
            "unsupervised_loader": train_loader,
            "supervised_loader": supervised_loader,
            "dual_loader": dual_loader
        }

        if args.cpn_initial == "means":
            model.task_initial(current_tasks=tasks[task_idx], means=cpn_means)
        else:
            model.task_initial(current_tasks=tasks[task_idx])
        trainer = pl.Trainer(
            gpus=num_gpus,
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
        model.train_loaders = train_loaders
        model.encoder = encoder
        # model.protoAug_start()
        trainer.fit(model, train_loaders, test_loader)
        wandb.finish()
        # model.protoAug_end()


if __name__ == '__main__':
    main()
