from torch import Tensor, device
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from metrics.base import Metrics


class Regression(Metrics):

    def __init__(self, num_classes: int, device: device):

        # TODO: problem is that we want to compute the metrics dimension-wise
        #   in case of multi-ouput regression (like QM9)
        # TODO: also gotta think about logging in this case

        super(Regression, self).__init__()

        self.mean_absolute_error = MeanAbsoluteError().to(device)
        self.mean_absolute_percentage_error = MeanAbsolutePercentageError().to(device)
        self.mean_squared_error = MeanSquaredError().to(device)

    def reset(self):

        self.mean_absolute_error.reset()
        self.mean_absolute_percentage_error.reset()
        self.mean_squared_error.reset()
    
    def compute_loss(self, input: Tensor, target: Tensor):

        input = input.reshape(target.shape)
        
        mse = self.mean_squared_error.forward(input, target)
        self.mean_absolute_error.update(input, target)
        self.mean_absolute_percentage_error.update(input, target)

        return mse
    
    def aggregate_metrics(self):

        mean_sq_error = self.mean_squared_error.compute().item()
        mean_abs_error = self.mean_absolute_error.compute().item()
        mean_abs_perc_error = self.mean_absolute_percentage_error.compute().item()

        self.reset()

        metrics = [
            ('Mean Squared Error', mean_sq_error),
            ('Mean Absolute Error', mean_abs_error),
            ('Mean Absolute Percentage Error', mean_abs_perc_error),
        ]

        return metrics