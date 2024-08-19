import numpy as np


def compute_metrics(output: np.ndarray,reference: np.ndarray):
    output = output.argmax(axis=-1)
    acc = (output == reference).sum().item() / len(reference)
    result_metric = {'Accuracy':acc}

    return result_metric

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
