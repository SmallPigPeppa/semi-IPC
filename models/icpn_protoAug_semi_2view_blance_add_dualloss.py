import random
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class IncrementalCPN(pl.LightningModule):
    def __init__(self, dim_feature, num_classes, pl_lambda, lr, epochs, warmup_epochs, **kwargs):
        super(IncrementalCPN, self).__init__()
        self.dim_feature = dim_feature
        self.num_calsses = num_classes
        self.pl_lambda = pl_lambda
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.extra_args = kwargs
        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.dim_feature)) for i in range(num_classes)])
        self.pl_lambda = 0.
        self.protoAug_lambda = 0.
        self.dual_lambda = 1.0

    def task_initial(self, current_tasks, means=None):
        if means is not None:
            for i in current_tasks:
                self.prototypes[i].data = torch.nn.Parameter((means[str(i)]).reshape(1, -1))
        no_grad_idx = [i for i in range(self.num_calsses) if i not in current_tasks]
        for i in no_grad_idx:
            self.prototypes[i].requires_grad = False
        for i in current_tasks:
            self.prototypes[i].requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)
        x = x.reshape(-1, 1, self.dim_feature)
        prototypes_list = [i for i in self.prototypes]
        d = torch.pow(x - torch.cat(prototypes_list), 2)
        d = torch.sum(d, dim=2)
        return d

    def training_step(self, batch, batch_idx):
        x, targets = batch['supervised_loader']
        d = self.forward(x)
        logits = -1. * d
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # pl loss
        pl_loss = torch.index_select(d, dim=1, index=targets)
        pl_loss = torch.diagonal(pl_loss)
        pl_loss = torch.mean(pl_loss)
        # all loss
        # loss = ce_loss + pl_loss * self.pl_lambda
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]

        if self.current_task_idx > 0:
            x_all = x
            y_all = targets
        else:
            x_all = x
            y_all = targets

        logits_all = -1. * self.forward(x_all)
        protoAug_loss = F.cross_entropy(logits_all, y_all)

        # loss = ce_loss + pl_loss * self.pl_lambda + protoAug_loss * self.protoAug_lambda
        # loss = ce_loss + pl_loss * self.pl_lambda

        # unlabel data
        x_unlabel, targets_unlabel = batch['unsupervised_loader']

        logits_unlabel = -1. * self(x_unlabel)
        probabilities_unlabel = F.softmax(logits_unlabel, dim=1)
        _, max_logits_unlabel = torch.max(logits_unlabel, dim=1)
        _, max_probabilities_unlabel = torch.max(probabilities_unlabel, dim=1)
        # mask = logits_unlabel[torch.arange(logits_unlabel.shape[0]), max_logits_unlabel] > 0.75
        mask = probabilities_unlabel[torch.arange(probabilities_unlabel.shape[0]), max_probabilities_unlabel] > 0.95

        # import pdb;pdb.set_trace()

        x_unlabel_high_conf = x_unlabel[mask]
        target_unlabel_high_conf = max_logits_unlabel[mask]
        target_unlabel_high_conf_2 = targets_unlabel[mask]
        # target_unlabel_high_conf = targets_unlabel[mask]

        # import pdb;pdb.set_trace()

        semi_x_all = torch.cat([x, x_unlabel_high_conf])
        semi_target_all = torch.cat([targets, target_unlabel_high_conf])
        semi_loss = F.cross_entropy(-1. * self(semi_x_all), semi_target_all)

        [x_weak, x_strong], targets_unlabel = batch['dual_loader']
        logits_weak = -1. * self(x_weak)
        probabilities_weak = F.softmax(logits_weak, dim=1)
        _, max_logits_weak = torch.max(logits_weak, dim=1)
        _, max_probabilities_weak = torch.max(probabilities_weak, dim=1)
        mask = probabilities_weak[torch.arange(probabilities_weak.shape[0]), max_probabilities_weak] > 0.95
        x_strong_high_conf = x_strong[mask]
        target_weak_high_conf = max_logits_weak[mask]
        semi_dual_loss = F.cross_entropy(-1. * self(x_strong_high_conf), target_weak_high_conf)

        # loss = ce_loss + pl_loss * self.pl_lambda + semi_loss

        # if semi_loss < 0.2:
        #     semi_loss = 0.
        # if pl_loss < 80:
        #     pl_loss = 0.
        # loss = pl_loss * self.pl_lambda + ce_loss + semi_loss
        # loss = pl_loss * self.pl_lambda + ce_loss
        loss = ce_loss + semi_dual_loss * self.dual_lambda + pl_loss * self.pl_lambda

        out = {"ce_loss": ce_loss, "pl_loss": pl_loss, "semi_loss": semi_loss, "semi_dual_loss": semi_dual_loss,
               "protoAug_loss": protoAug_loss,
               "acc": acc, "loss": loss}
        log_dict = {"train_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        d = self.forward(x)
        logits = -1. * d
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # pl loss
        pl_loss = torch.index_select(d, dim=1, index=targets)
        pl_loss = torch.diagonal(pl_loss)
        pl_loss = torch.mean(pl_loss)

        # preds
        preds = torch.argmax(logits, dim=1)

        # acc all
        acc = torch.sum(preds == targets) / targets.shape[0]

        # 初始化准确率字典
        task_accuracies = {}

        # 对于当前任务及之前的任务，计算准确率
        for past_task_idx in range(self.current_task_idx + 1):
            task_classes = self.tasks[past_task_idx].to(self.device)
            task_mask = torch.isin(targets, task_classes)

            if task_mask.any():
                task_targets = targets[task_mask]
                task_preds = preds[task_mask]
                task_acc = torch.sum(task_preds == task_targets).float() / task_targets.shape[0]
                task_accuracies[f"task_{past_task_idx}_acc"] = task_acc
            else:
                task_accuracies[f"task_{past_task_idx}_acc"] = torch.tensor(0.)

        # all loss
        loss = ce_loss + pl_loss * self.pl_lambda

        protoAug_loss = 0.

        # 汇总输出
        out = {"ce_loss": ce_loss, "pl_loss": pl_loss, 'protoAug_loss': protoAug_loss, "loss": loss, "acc": acc}
        out.update(task_accuracies)

        # 日志
        log_dict = {"val_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def protoAug_start(self):
        # self.radius = 0.1
        self.old_classes = []
        for task in self.tasks[:self.current_task_idx]:
            self.old_classes.extend(task.tolist())
        self.new_classes = self.tasks[self.current_task_idx]
        self.protoAug_lambda = 1.0

    def protoAug_end(self):
        # Calculate the mean and variance of each class in self.new_classes
        class_means = {}
        class_features = {}
        self.eval()
        for x, targets in self.train_loaders['supervised_loader']:
            targets = targets.to(self.device)
            inputs = x.to(self.device)
            for class_id in self.new_classes:
                # Skip if the class_id is not in targets
                if not (targets == class_id).any():
                    continue
                class_id = class_id.item()
                indices = (targets == class_id)
                features = inputs[indices]
                # If class_id is encountered for the first time, initialize mean and features list
                if class_id not in class_means:
                    class_means[class_id] = features.mean(dim=0, keepdim=True)
                    class_features[class_id] = [features]
                # If class_id has been encountered before, update mean and append features
                else:
                    class_means[class_id] = (class_means[class_id] * len(class_features[class_id]) + features.mean(
                        dim=0, keepdim=True)) / (len(class_features[class_id]) + 1)
                    class_features[class_id].append(features)

        if self.current_task_idx == 0:
            # Compute average radius
            radii = []
            for class_id in class_features:
                features = torch.cat(class_features[class_id], dim=0)
                # Here, replace the class_means with self.prototypes[class_id]
                features = features - class_means[class_id]
                # features = features - self.prototypes[class_id]
                cov = torch.matmul(features.t(), features) / features.shape[0]
                radius = torch.trace(cov) / features.shape[1]
                radii.append(radius)
            avg_radius = torch.sqrt(torch.mean(torch.stack(radii)))

            # Store average radius
            self.radius = avg_radius
