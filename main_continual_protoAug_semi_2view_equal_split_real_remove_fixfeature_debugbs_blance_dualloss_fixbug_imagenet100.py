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
from models.icpn_protoAug_semi_2view_blance_add_dualloss_imagenet100 import IncrementalCPN
from collections import defaultdict
import random
from torch.utils.data import Subset, Dataset
from tqdm import tqdm


# def keep_n_samples_per_class(dataset, n, return_means=False):
#     # 首先，提取所有的标签
#     if hasattr(dataset, 'targets'):
#         labels = dataset.targets
#     else:
#         labels = [label for _, label in dataset]
#
#     # 转换成tensor以加快处理速度
#     if not isinstance(labels, torch.Tensor):
#         labels = torch.tensor(labels)
#
#     # 为每个类别收集样本索引
#     class_samples = defaultdict(list)
#     for label in torch.unique(labels):
#         label_indices = (labels == label).nonzero(as_tuple=True)[0]
#         class_samples[label.item()] = label_indices.tolist()
#
#     new_indices = []
#
#     # 为每个类别保留n个样本
#     for label, samples in class_samples.items():
#         if len(samples) <= n:
#             new_indices.extend(samples)
#         else:
#             new_indices.extend(random.sample(samples, n))
#
#     # 创建一个新的数据集
#     new_dataset = Subset(dataset, new_indices)
#
#     # 如果需要返回每类的均值
#     if return_means:
#         class_means = {}
#         for label in class_samples.keys():
#             selected_samples = [dataset[i][0] for i in new_indices if dataset[i][1] == label]
#             class_means[str(torch.tensor(label))] = torch.mean(torch.stack(selected_samples), dim=0)
#
#         return new_dataset, class_means
#     else:
#         return new_dataset


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


class SubsetWithReplacement(Dataset):
    """A subset of a dataset at specified indices, with replacement."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def keep_n_samples_per_class(dataset, n=10):
    try:
        # 尝试直接获取 targets（对于未被 split_dataset 处理的数据集）
        targets = dataset.targets
    except AttributeError:
        # 对于经过 split_dataset 处理的数据集，需要提取 targets
        targets = [dataset.dataset.targets[i] for i in dataset.indices]

    classes = set(targets)

    # 为每个类别选择 n 个样本
    selected_indices = []
    for cls in classes:
        cls_indices = [i for i, target in enumerate(targets) if target == cls]
        selected_indices.extend(random.sample(cls_indices, min(n, len(cls_indices))))

    # 重复选取样本直到达到原始数据集大小
    multiplier = len(targets) // len(selected_indices)
    additional_indices = random.choices(selected_indices, k=(len(targets) - multiplier * len(selected_indices)))

    final_indices = selected_indices * multiplier + additional_indices
    random.shuffle(final_indices)  # 打乱顺序以增加随机性

    # 处理 split_dataset 返回的 Subset 数据集
    if isinstance(dataset, torch.utils.data.Subset):
        # 将选定的索引转换回原始数据集的索引
        final_indices = [dataset.indices[idx] for idx in final_indices]
        return torch.utils.data.Subset(dataset.dataset, final_indices)
    else:
        return SubsetWithReplacement(dataset, final_indices)


def compute_class_means(dataset, encoder, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 确保编码器处于评估模式
    encoder.eval()
    encoder.to(device)

    # 存储每个类别的特征
    class_samples = {}


    for inputs, labels in tqdm(dataloader,desc="compute means"):
        # 使用编码器处理数据
        with torch.no_grad():
            features = encoder(inputs.to(device))
            for feature, label in zip(features, labels):
                label = label.item()
                if label in class_samples:
                    class_samples[label].append(feature)
                else:
                    class_samples[label] = [feature]

    # 计算每个类别的特征均值
    class_means = {}
    for label, features in class_samples.items():
        class_means[str(torch.tensor(label))] = torch.mean(torch.stack(features), dim=0)

    return class_means


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
    # train_dataset = keep_n_samples_per_class(train_dataset, n=10)
    dual_dataset = get_dual_dataset(dataset=args.dataset, data_path=args.data_path)

    for task_idx in range(0, args.num_tasks):
        model.tasks = tasks
        model.current_task_idx = task_idx
        # model.batch_size = 64
        # model.semi_batch_size = 64
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
            task_idx=list(range(task_idx + 1)),
        )
        dual_dataset_task = split_dataset(
            dual_dataset,
            tasks=tasks,
            task_idx=[task_idx],
        )
        print("finished...")
        print('model.device:', model.device)
        # train_dataset_task_fix, test_dataset_task_fix, cpn_means = get_pretrained_dataset(
        #     encoder=encoder,
        #     train_dataset=train_dataset_task,
        #     test_dataset=test_dataset_task,
        #     return_means=True)

        # train_loader = DataLoader(train_dataset_task, batch_size=64, shuffle=True)
        # test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=True)

        new_dataset = keep_n_samples_per_class(train_dataset_task, n=10)
        cpn_means = compute_class_means(new_dataset, encoder, batch_size=512)
        bs=64
        # train_loader = DataLoader(train_dataset_task, batch_size=bs, shuffle=True, pin_memory=True, num_workers=8)
        # dual_loader = DataLoader(dual_dataset_task, batch_size=bs, shuffle=True, pin_memory=True, num_workers=8)
        # new_loader = DataLoader(new_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=8)
        # test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)

        train_loader = DataLoader(train_dataset_task, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
        dual_loader = DataLoader(dual_dataset_task, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
        new_loader = DataLoader(new_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)


        print("keep_n_samples_per_class...")
        # _, cpn_means = keep_n_samples_per_class(train_dataset_task_fix, n=10, return_means=True)
        # supervised_data = keep_n_samples_per_class(train_dataset_task, n=10, return_means=False)
        print("finished...")
        # supervised_loader2 = DataLoader(supervised_data, batch_size=64, shuffle=True,pin_memory=True,num_workers=8)

        train_loaders = {
            "supervised_loader": train_loader,
            "dual_loader": dual_loader,
            "new_loader": new_loader,
        }

        if args.cpn_initial == "means":
            model.task_initial(current_tasks=tasks[task_idx], means=cpn_means)
        else:
            model.task_initial(current_tasks=tasks[task_idx])
        num_gpus = 2
        trainer = pl.Trainer(
            # accelerator='gpu',
            gpus=num_gpus,
            # strategy='ddp',
            max_epochs=args.epochs,
            accumulate_grad_batches=1,
            # gradient_clip_val=1.0,
            sync_batchnorm=True,
            # accelerator='gpu',
            logger=wandb_logger,
            # checkpoint_callback=False,
            enable_checkpointing=None,
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
