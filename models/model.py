import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    def forward(self, x, y, D=None):
        a = torch.exp(-torch.mul(x, y))
        return a.mul(D).sum()*(self.args.train_dataset_len/x.shape[0])


class BMM_v1(nn.Module):
    def __init__(self, num_classes, feature_dim):
        """
        :param num_classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        """
        super(BMM_v1, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, feature_dim//2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim//2, 10)
        self.act2 = nn.ReLU()
        # self.dp = nn.Dropout(p=0.5),
        self.fc3 = nn.Linear(10, 1)
        self.act3 = nn.Sigmoid()
        # self.fc2 = nn.Linear(10, 2)
        # self.act2 = nn.Softmax()

    def forward(self, x):
        y = self.act1(self.fc1(x))
        y = self.act2(self.fc2(y))
        y = self.act3(self.fc3(y))
        # print(y.shape)
        # print("before change:", y)
        y = y.squeeze(1)
        if self.num_classes == 2:  # 输出为-1, 1
            y = 2 * (y - 0.5)  # 将输出为(0, 1)的转换为(-1, 1)
        # print("after change:", y)
        return y