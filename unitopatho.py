import torch
import torchvision
import numpy as np
import cv2
import os

class UTP(torch.utils.data.Dataset):
    def __init__(self, df, T, path, target, subsample=-1, gray=False, mock=False):
        self.path = path
        self.df = df
        self.T = T
        self.target = target
        self.subsample = subsample
        self.mock = mock
        self.gray = gray

        allowed_target = ['type', 'grade', 'top_label']
        if target not in allowed_target:
            print(f'Target must be in {allowed_target}, got {target}')
            exit(1)

        print(f'Loaded {len(self.df)} images')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        image_id = entry.image_id
        image_id = os.path.join(self.path, entry.top_label_name, image_id)

        img = None

        if self.mock:
            C = 1 if self.gray else 3
            img = np.random.randint(0, 255, (224, 224, C)).astype(np.uint8)

        else:
            img = cv2.imread(image_id)
            if self.subsample != -1:
                w = img.shape[0]
                while w//2 > self.subsample:
                    img = cv2.resize(img, (w//2, w//2))
                    w = w//2
                img = cv2.resize(img, (self.subsample, self.subsample))

            if self.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.T is not None:
            img = self.T(img)

        return img, entry[self.target]
