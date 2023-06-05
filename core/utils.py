# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import logging
from core.param import *
import numpy as np

class Distribution(object):
    def __init__(self, num_sample, logger=None):
        super().__init__()
        self.num_sample = num_sample
        self.logger = logger
        self.weights = np.array([1.0 / num_sample for _ in range(num_sample)])

    def norm(self, sum_num=1.0):
        weights = self.weights / sum(self.weights) * sum_num
        self.weights = weights
        return self.weights

    def mul(self, other):
        assert self.num_sample == other.num_sample
        weights = self.weights * other.weights  # element-wise
        self.weights = weights
        # self.show()
        self.weights = self.norm()
        return self.weights
    
    def show(self, name=None):
        self.logger.info(f'Distribution: {name}')
        self.logger.info(self.weights)

    def paint(self, targetcollect, name=None, save_pth=None):
        min = np.min(self.weights)
        max = np.max(self.weights)
        mean = np.mean(self.weights)
        std = np.std(self.weights)
        print("min, max, mean, std of weights: ", min, max, mean, std)
        import matplotlib.pyplot as plt
        tmpweights = self.weights
        tmpweights = (tmpweights - mean) / (max - min)
        color_list = ['red' if targetcollect[i] == 1 else 'blue' for i in range(self.num_sample)]
        plt.bar(range(self.num_sample), tmpweights, color=color_list)
        plt.title(name)
        plt.savefig(save_pth)
        plt.show()
        return plt

def accuracy_twoclass(output, target):
    output_ = output.detach().cpu().numpy()
    target_ = target.detach().cpu().numpy()
    for i in range(len(output_)):
        output_[i] = -1 if output_[i] < 0 else 1
    acc = np.sum(output_ == target_) / len(output_)
    return acc


def create_logging(log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger()

    handlers = []
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger

def save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='Latest'):
    model = model.module if hasattr(model, 'module') else model
    save_state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'max_accuracy': max_accuracy
    }
    # 保存最新(好)epoch参数
    lastest_save_path = os.path.join(args.path_log, '%s-%s.pth' % (args.name, save_name))
    torch.save(save_state, lastest_save_path)
    logger.info(f"{lastest_save_path} saved !!!")