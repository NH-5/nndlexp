"""NLLLoss module."""

import torch.nn as nn

# NLLLoss function输入是一个对数概率向量和一个目标标签。NLLLoss() ，即负对数似然损失函数（Negative Log Likelihood）
class NLLLoss(nn.Module):
    """Negative log-likelihood loss."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self._loss = nn.NLLLoss(reduction=reduction)

    def forward(self, logits, label):
        return self._loss(logits, label)

    def construct(self, logits, label):
        return self.forward(logits, label)
