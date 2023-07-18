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

        self.protoAug_lambda = 1.0

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
        x = x.reshape(-1, 1, self.dim_feature)
        prototypes_list = [i for i in self.prototypes]
        d = torch.pow(x - torch.cat(prototypes_list), 2)
        d = torch.sum(d, dim=2)
        return d

    def share_step(self, batch, batch_idx):
        x, targets = batch
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
            old_classes = self.old_classes
            radius = self.radius
            prototypes = self.prototypes
            batch_size = self.semi_batch_size
            batchsize_new = batch_size // 2
            batchsize_old = batch_size // 2

            # x_new, y_new = batch["semi_data"]
            x_new = x[:batchsize_new]
            y_new = targets[:batchsize_new]


            # import pdb;pdb.set_trace()
            y_old = torch.tensor(random.choices(old_classes, k=batch_size))[:batchsize_old].to(self.device)
            # Convert old_y to Python list
            y_old_list = y_old.tolist()
            # Index prototype with old_y_list
            prototype_old = torch.cat([prototypes[i] for i in y_old_list])
            x_old = prototype_old + torch.randn(batchsize_old, self.dim_feature).to(self.device) * radius

            y_all = torch.cat([y_new, y_old], dim=0)
            x_all = torch.cat([x_new, x_old], dim=0)
        else:
            x_all = x
            y_all = targets

        logits_all = -1. * self.forward(x_all)
        protoAug_loss = F.cross_entropy(logits_all, y_all)

        loss = ce_loss + pl_loss * self.pl_lambda + protoAug_loss * self.protoAug_lambda

        return {"ce_loss": ce_loss, "pl_loss": pl_loss, 'protoAug_loss': protoAug_loss, "acc": acc, "loss": loss}

    def training_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"train_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def validation_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"val_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def test_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"test_" + k: v for k, v in out.items()}
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
        for x, targets in self.train_loader:
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

