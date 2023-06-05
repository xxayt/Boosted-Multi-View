import torch
import numpy as np
import random
import scipy.io as scio

class Multi_view_data(object):
    """ Dataloader for Caltech101 datasets """
    def __init__(self, data_name='Caltech101-7', mode='train', TwoClass=False, Balance=False):
        '''
        param: data_name: 数据集名称
        param: mode: train / val / test
        param: TwoClass: 二分类问题(取前两类,保证两类样本个数相同)
        '''
        super(Multi_view_data, self).__init__()
        data_dir = './datasets/' + data_name + '.mat'
        mat = scio.loadmat(data_dir)  # 读取mat文件
        X = np.squeeze(mat['X'])  # 6*sample_num*dim. sample_num=1474, 2386. dim=(48, 40, 254, 1984, 512, 928)
        Y = np.squeeze(mat['Y'])  # 1474*1. (1类为0~434, 2类为435~1232, 3类为1233~1284, 4类为1285~1318, 5类为1319~1353, 6类为1354~1417, 7类为1418~1473)
        sample_class_num = np.squeeze(mat['lenSmp']).tolist()  # sample_class_num: 7*1. (435, 798, 52, 34, 35, 64, 56)
        feature_view = np.squeeze(mat['feanames']).tolist()  # 6*1. ['gabor', 'wavelet_moments', 'cenhist', 'hog', 'gist', 'lbp']   
        # print(X.shape)  # (6, )
        # print(X[0].shape)  # (1474, 48)
        # print(Y.shape)  # (1474,)
        
        # Store the index of samples into the data_list
        X_list = X.tolist()
        Y_list = Y.tolist()
        data_list = []
        class_index_start = 0
        class_index_end = 0
        if TwoClass:  # 二分类问题
            sample_class_num = sample_class_num[:2]
        for iter, class_num in enumerate(sample_class_num):
            # print('The %d-th class: %d' % (iter, class_num))
            class_index_end += class_num
            sample_index = range(class_index_start, class_index_end)  # 确定类别iter的索引范围
            if TwoClass:
                target = 1 if iter == 0 else -1
            else:
                target = Y_list[class_index_start]

            class_samples = []
            for i in range(len(sample_index)):
                sample_view_all = np.tile(sample_index[i], len(feature_view))  # 沿着列方向复制. 构建6*1的向量
                class_samples.append((sample_view_all, target))  # ([0 0 0 0 0 0], 1)
            # 按照7:2:1比例拆分为train, val, test
            random.seed(int(200))
            if Balance:
                every_class_num = min(sample_class_num[0], sample_class_num[1])
                train_index = random.sample(range(0, class_num), int(0.7*every_class_num))
                rem_index = [rem for rem in range(0, class_num) if rem not in train_index]
                val_index = random.sample(rem_index, int(2/3.0*(every_class_num - int(0.7*every_class_num))))
                rem_index = [rem for rem in rem_index if rem not in val_index]
                test_index = random.sample(rem_index, every_class_num - int(0.7*every_class_num) - int(2/3.0*(every_class_num - int(0.7*every_class_num))))
            else:
                train_index = random.sample(range(0, class_num), int(0.7*class_num))
                rem_index = [rem for rem in range(0, class_num) if rem not in train_index]
                val_index = random.sample(rem_index, int(2/3.0*len(rem_index)))
                test_index = [rem for rem in rem_index if rem not in val_index]

            # 只存每个feature(view)的index和label
            train_part = [class_samples[i] for i in train_index]
            val_part = [class_samples[i] for i in val_index]
            test_part = [class_samples[i] for i in test_index]
            # 按照mode存储
            if mode == 'train':
                data_list.extend(train_part)
            elif mode == 'val':
                data_list.extend(val_part)
            else:
                data_list.extend(test_part)
            # 更新class_index_start
            class_index_start = class_index_end
        
        self.data_list = data_list   
        self.X_list = X_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """ Load an episode each time """
        X_list = self.X_list
        (sample_view_all, target) = self.data_list[index]    
        Sample_Fea_Allviews = []

        for i in range(len(sample_view_all)):      
            sample_temp = np.array(X_list[i][sample_view_all[i]])
            sample_temp = sample_temp.astype(float)
            sample_temp = torch.from_numpy(sample_temp)
            Sample_Fea_Allviews.append(sample_temp.type(torch.FloatTensor))
        return (Sample_Fea_Allviews, target)

# t = Multi_view_data(data_name='Caltech101-7', mode='train', TwoClass=True)
# print(len(t))  # 435*0.7=304, 798*0.7=558. 304+558=862
# t = Multi_view_data(data_name='Caltech101-7', mode='val', TwoClass=True)
# print(len(t))  # 131*2/3=87, 240*2/3=160, 87+160=247
# t = Multi_view_data(data_name='Caltech101-7', mode='test', TwoClass=True)
# print(len(t))  # 44+80=124

def get_loader(data_name='Caltech101-7', mode='train', TwoClass=False, args=None):
    print('Choose TwoClass: ', TwoClass)
    print('mode: ', mode)
    dataset = Multi_view_data(data_name, mode, TwoClass)
    if mode == 'train':
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers)
    print('dataset len: ', len(dataset))  # 862, 247, 124
    print('data_loader len: ', len(data_loader))  # 27, 8, 4
    return data_loader, len(dataset)

