import random
import torch
from PIL import Image
import os
import pandas as pd


class Waterbird(torch.utils.data.Dataset):
    def __init__(self, root, df, transform, train=True, mode='bg_all'):
        self.transform = transform
        self.train = train
        self.df = df

        cols = self.df.columns
        arr = self.df.to_numpy()
        ## random shuffle arr using seed 10
        random.seed(10)
        random.shuffle(arr)
        new_arr = []
        wbg = 0
        lbg = 0
        for item in arr:
            if item[2] == 0:
                if item[4] == 0 and lbg < 3500:
                    lbg += 1
                    new_arr.append(item)
                elif item[4] == 1 and wbg < 100:
                    wbg += 1
                    new_arr.append(item)
        self.df = pd.DataFrame(new_arr, columns=cols)
        ## save the new metadata
        self.df.to_csv(f'new_df.csv', index=False)

        wb_on_l = self.df[(self.df['y'] == 0) & (self.df['place'] == 0)]
        wb_on_w = self.df[(self.df['y'] == 0) & (self.df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = wb_on_l.iloc[:]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:])
        normal_df = wb_on_w.iloc[:]
        normal_df_np = normal_df['img_filename'].to_numpy()
        copy_count = 1
        for _ in range(copy_count):
            self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:])

        if train:
            self.image_paths = self.normal_paths
            print('here', len(self.image_paths), len(self.df), len(wb_on_w), len(wb_on_l))
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        mask = torch.zeros([1, *img.size()[1:]])

        return {
            'image': img,
            "mask": mask,
            "is_anomaly": self.labels[idx],
            "image_path": self.image_paths[idx],
        }
