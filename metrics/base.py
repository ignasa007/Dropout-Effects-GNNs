from torch import Tensor


class Metrics:

    def __init__(self):

        pass

    def reset(self):

        raise NotImplementedError
    
    def compute_loss(self, input: Tensor, target: Tensor):

        raise NotImplementedError
    
    def aggregate_metrics(self):

        raise NotImplementedError