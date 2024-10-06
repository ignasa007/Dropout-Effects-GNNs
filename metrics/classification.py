import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC, \
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC

from metrics.base import Metrics


class Classification(Metrics):

    def __init__(self, num_classes: int):

        super(Classification, self).__init__()

        if not isinstance(num_classes, int):
            raise TypeError(f'Expected `num_classes` to be an instance of `int` (got {type(num_classes)}).')
        
        if num_classes == 2:
            self.nonlinearity = torch.sigmoid
            self.loss_fn = lambda input, target: BCEWithLogitsLoss(reduction='sum')(input, target.float())
            self.accuracy_fn = BinaryAccuracy()
            self.f1score_fn = BinaryF1Score()
            self.auroc_fn = BinaryAUROC()
        elif num_classes > 2:
            self.nonlinearity = lambda probs: torch.softmax(probs, dim=-1)
            self.loss_fn = CrossEntropyLoss(reduction='sum')
            self.accuracy_fn = MulticlassAccuracy(num_classes)
            self.f1score_fn = MulticlassF1Score(num_classes)
            self.auroc_fn = MulticlassAUROC(num_classes)
        else:
            raise ValueError(f'Expected `num_classes` to be >1 (got {num_classes}).')
        
        self.reset()
        
    def reset(self):

        self.n_samples = self.total_ce_loss = 0
        self.accuracy_fn.reset()
        self.f1score_fn.reset()
        self.auroc_fn.reset()
    
    def compute_loss(self, input: Tensor, target: Tensor):

        # squeeze to make the input compatible for BCE loss
        input = input.squeeze()

        # loss expects logits, but not probabilities
        batch_ce_loss = self.loss_fn(input, target)
        self.total_ce_loss += batch_ce_loss.item()
        self.n_samples += target.size(0)

        # other metrics expect (rather, can work with) probabilities, but not logits
        preds = self.nonlinearity(input)
        self.accuracy_fn.update(preds, target)
        self.f1score_fn.update(preds, target)
        self.auroc_fn.update(preds, target)

        return batch_ce_loss / target.size(0)

    def compute_metrics(self):

        cross_entropy = self.total_ce_loss / self.n_samples
        accuracy = self.accuracy_fn.compute().item()
        f1_score = self.f1score_fn.compute().item()
        auroc = self.auroc_fn.compute().item()

        self.reset()

        metrics = [
            ('Cross Entropy Loss', cross_entropy),
            ('Accuracy', accuracy),
            ('F1 Score', f1_score),
            ('AU-ROC', auroc),
        ]

        return metrics