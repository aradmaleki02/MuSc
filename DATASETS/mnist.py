import pickle

import torch
from torch.utils.data import Dataset


class MNIST_Dataset(Dataset):
    def __init__(self, train, test_id=1, transform=None):
        self.train = train
        self.transform = transform
        if train:
            with open('./content/mnist_shifted_dataset/train_normal.pkl', 'rb') as f:
                normal_train = pickle.load(f)
            self.images = normal_train['images']
            self.labels = [0]*len(self.images)
        else:
            if test_id == 1:
                with open('./content/mnist_shifted_dataset/test_normal_main.pkl', 'rb') as f:
                    normal_test = pickle.load(f)
                with open('./content/mnist_shifted_dataset/test_abnormal_main.pkl', 'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0]*len(normal_test['images']) + [1]*len(abnormal_test['images'])
            else:
                with open('./content/mnist_shifted_dataset/test_normal_shifted.pkl', 'rb') as f:
                    normal_test = pickle.load(f)
                with open('./content/mnist_shifted_dataset/test_abnormal_shifted.pkl', 'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0]*len(normal_test['images']) + [1]*len(abnormal_test['images'])

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        ret = {
            'image': image,
            'is_anomaly': self.labels[index],
        }
        return ret

    def __len__(self):
        return len(self.images)
