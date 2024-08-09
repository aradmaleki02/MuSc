import torch
from glob import glob
from PIL import Image
import random


class BrainTest(torch.utils.data.Dataset):
    def __init__(self, transform, test_id=1):

        self.transform = transform
        self.test_id = test_id

        test_normal_path = glob('./Br35H/dataset/test/normal/*')
        test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

        random.shuffle(test_anomaly_path)
        random.shuffle(test_normal_path)
        # test_anomaly_path = test_anomaly_path[:len(test_anomaly_path) // 5]
        # test_normal_path = test_normal_path[:len(test_normal_path) // 5]

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob('./brats/dataset/test/normal/*')
            test_anomaly_path = glob('./brats/dataset/test/anomaly/*')

            random.shuffle(test_anomaly_path)
            random.shuffle(test_normal_path)
            test_anomaly_path = test_anomaly_path[:len(test_anomaly_path) // 5]
            test_normal_path = test_normal_path[:len(test_normal_path) // 5]

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)

        has_anomaly = 0 if self.test_label[idx] == 0 else 1

        mask = torch.zeros([1, *image.size()[1:]])

        ret = {
            'image': image,
            # "mask": None,
            "is_anomaly": has_anomaly,
            # "image_path": None,
        }

        return ret
