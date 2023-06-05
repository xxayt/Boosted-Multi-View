import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    def forward(self, x, y):
        return torch.exp(-torch.mul(x, y)).mean()


class Baseline_Concat(nn.Module):
    def __init__(self, num_classes, feature_dim):
        """
        :param num_classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        """
        super(Baseline_Concat, self).__init__()
        self.num_classes = num_classes
        self.features_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, feature_dim//2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim//2, 10)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        # concat
        # x = torch.cat(x, dim=1)  # [32, 3766]
        # test dim_num & lr
        cenhist_list =  [x[2] for _ in range(12)]
        x = torch.cat(cenhist_list, dim=1)  # [32, 3048]
        # gabor_list =  [x[0] for _ in range(60)]
        # x = torch.cat(gabor_list, dim=1)  # [32, 2880]

        y = self.act1(self.fc1(x))
        y = self.act2(self.fc2(y))
        y = self.act3(self.fc3(y))
        y = y.squeeze(1)
        if self.num_classes == 2:  # 输出为-1, 1
            y = 2 * (y - 0.5)  # 将输出为(0, 1)的转换为(-1, 1)
        return y


class Baseline_DecisionFusion(nn.Module):
    def __init__(self, num_classes, feature_dict):
        """
        :param num_classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        """
        super(Baseline_DecisionFusion, self).__init__()
        self.num_classes = num_classes
        self.feature_dict = feature_dict
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dict[view], self.feature_dict[view]//2),
                nn.ReLU(),
                nn.Linear(self.feature_dict[view]//2, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            ) for view in self.feature_dict.keys()
        ])

    def forward(self, x):
        y = [self.classifier[i](x[i]) for i in range(len(x))]
        y = [y[i].squeeze(1) for i in range(len(y))]
        y = torch.mean(torch.stack(y), dim=0)  # 决策融合(简单平均法)
        if self.num_classes == 2:  # 输出为-1, 1
            y = 2 * (y - 0.5)  # 将输出为(0, 1)的转换为(-1, 1)
        return y