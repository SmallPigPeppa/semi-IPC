import random
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm


class mFC(pl.LightningModule):
    def __init__(self, dim_feature, num_classes, pl_lambda, lr, epochs, warmup_epochs, **kwargs):
        super(mFC, self).__init__()
        self.dim_feature = dim_feature
        self.num_calsses = num_classes
        self.pl_lambda = pl_lambda
        self.pa_lambda = 1.0
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.extra_args = kwargs
        self.fc = nn.Linear(in_features=self.dim_feature, out_features=num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        # self.encoder.eval()
        # with torch.no_grad():
        x = self.encoder(x)
        logits = self.fc(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        # ce loss
        loss = F.cross_entropy(logits, targets)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        out = {"acc": acc,
               "loss": loss}
        log_dict = {"train_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        # ce loss
        loss = F.cross_entropy(logits, targets)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        out = {"acc": acc, "loss": loss}
        log_dict = {"val_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

