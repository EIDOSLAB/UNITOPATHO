#!/usr/bin/env python3

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import albumentations
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2

import numpy as np
import pandas as pd
import cv2
import os
import wandb
import copy

import unitopatho
import utils

import re
import argparse

import torchstain

from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from functools import partial
from multiprocessing import Manager

torch.multiprocessing.set_sharing_strategy('file_system')
manager = Manager()


def resnet18(n_classes=2):
    model = torchvision.models.resnet18(pretrained='imagenet')
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
    return model

def preprocess_df(df, label):
    if label == 'norm':
        df.loc[df.grade == 0, 'grade'] = -1
        df.loc[df.type == 'norm', 'grade'] = 0

    df = df[df.grade >= 0].copy()

    if label != 'both' and label != 'norm':
        df = df[df.type == label].copy()
    return df

def main(config):
    checkpoint = None
    if config.test is not None:
        print('=> Loading saved checkpoint')
        checkpoint = torch.hub.load_state_dict_from_url(f'https://api.wandb.ai/files/eidos/UnitoPath-v1/{config.test}/model.pt', 
                                                        map_location='cpu', progress=True, check_hash=False)
        test = config.test
        device = config.device
        p = config.path
        config = checkpoint['config']
        config.test = test
        config.device = device
        config.path = p

    utils.set_seed(config.seed)
    scaler = torch.cuda.amp.GradScaler()

    if config.test is None:
        wandb.init(config=config,
                    project=f'unitopatho')

    path = os.path.join(config.path, str(config.size))
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))

    groupby = config.target + ''
    print('=> Raw data (train)')
    print(train_df.groupby(groupby).count())

    print('\n=> Raw data (test)')
    print(test_df.groupby(groupby).count())

    if config.target == 'grade':
        train_df = preprocess_df(train_df, config.label)
        test_df = preprocess_df(test_df, config.label)

        # balance train_df (sample mean size)
        groups = train_df.groupby('grade').count()
        grade_min = int(groups.image_id.idxmin())
        mean_size = int(train_df.groupby('grade').count().mean()['image_id'])

        train_df = pd.concat((
            train_df[train_df.grade == 0].sample(mean_size, replace=(grade_min==0), random_state=config.seed).copy(),
            train_df[train_df.grade == 1].sample(mean_size, replace=(grade_min==1), random_state=config.seed).copy()
        ))

    else:
        # balance train_df (sample 3rd min_size)
        min_size = np.sort(train_df.groupby(groupby).count()['image_id'])[2]
        train_df = train_df.groupby(groupby).apply(lambda group: group.sample(min_size, replace=len(group) < min_size, random_state=config.seed)).reset_index(drop=True)

    print('\n---- DATA SUMMARY ----')
    print('---------------------------------- Train ----------------------------------')
    print(train_df.groupby(groupby).count())
    print(len(train_df.wsi.unique()), 'WSIs')

    print('\n---------------------------------- Test ----------------------------------')
    print(test_df.groupby(groupby).count())
    print(len(test_df.wsi.unique()), 'WSIs')

    im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet
    norm = dict(
        rgb=dict(mean=im_mean,
                    std=im_std),
        he=dict(mean=im_mean,
                std=im_std),
        gray=dict(mean=[0.5],
                    std=[1.0])
    )

    T_aug = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(90, p=0.5)
    ])
    T_jitter = albumentations.ColorJitter()

    mean, std = norm[config.preprocess]['mean'], norm[config.preprocess]['std']
    print('=> mean, std:', mean, std)
    T_tensor = ToTensorV2()
    T_post = albumentations.Compose([
        albumentations.Normalize(mean, std),
        T_tensor
    ])

    print('=> Preparing stain normalizer..')
    he_target = cv2.cvtColor(cv2.imread('data/target.jpg'), cv2.COLOR_BGR2RGB)
    normalizer = torchstain.MacenkoNormalizer(backend='torch')
    normalizer.fit(T_tensor(image=he_target)['image']*255)
    print('=> Done')

    def normalize_he(x):
        if config.preprocess == 'he':
            img = x
            try:
                img = T_tensor(image=img)['image']*255
                img, _, _ = normalizer.normalize(img, stains=False)
                img = img.numpy().astype(np.uint8)
            except Exception as e:
                print('Could not normalize image:', e)
                img = x
            return img
        return x

    def apply_transforms(train, img):
        img = normalize_he(img)
        if train:
            img = T_aug(image=img)['image']
            if config.preprocess == 'rgb':
                img = T_jitter(image=img)['image']
        x = img
        return T_post(image=x)['image']

    T_train = partial(apply_transforms, True)
    T_test = partial(apply_transforms, False)

    datasets_kwargs = {
        'path': path,
        'subsample': config.subsample,
        'target': config.target,
        'gray': config.preprocess == 'gray',
        'mock': config.mock
    }

    train_dataset = unitopatho.UTP(train_df, T=T_train, **datasets_kwargs)
    test_dataset = unitopatho.UTP(test_df, T=T_test, **datasets_kwargs)

    # Final loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, 
                                               batch_size=config.batch_size, 
                                               num_workers=config.n_workers, 
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, 
                                              batch_size=config.batch_size, 
                                              num_workers=config.n_workers, 
                                              pin_memory=True)

    n_classes = len(train_df[config.target].unique())
    print(f'=> Training for {n_classes} classes')

    n_channels = {
        'rgb': 3,
        'he': 3,
        'gray': 1
    }

    model = resnet18(n_classes=n_classes)
    model.conv1 = torch.nn.Conv2d(n_channels[config.preprocess], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    model = model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = F.cross_entropy

    for epoch in range(config.epochs):
        if config.test is None:
            train_metrics = utils.train(model, train_loader, criterion,
                                        optimizer, config.device, metrics=utils.metrics,
                                        accumulation_steps=config.accumulation_steps, scaler=scaler)
            scheduler.step()

        test_metrics = utils.test(model, test_loader, criterion, config.device, metrics=utils.metrics)

        if config.test is None:
            print(f'Epoch {epoch}: train: {train_metrics}')
            wandb.log({'train': train_metrics,
                       'test': test_metrics})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config},
                    os.path.join(wandb.run.dir, 'model.pt'))
        
        print(f'test: {test_metrics}')
        if config.test is not None:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--path', default=f'{os.path.expanduser("~")}/data/UNITOPATHO', type=str, help='UNITOPATHO dataset path')
    parser.add_argument('--size', default=100, type=int, help='patch size in µm (default 100)')
    parser.add_argument('--subsample', default=-1, type=int, help='subsample size for data (-1 to disable, default -1)')

    # optimizer & network config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--n_workers', default=8, type=int)
    parser.add_argument('--architecture', default='resnet18', help='resnet18, resnet50, densenet121')

    # training config
    parser.add_argument('--preprocess', default='rgb', help='preprocessing type, rgb, he or gray. Default: rgb')
    parser.add_argument('--target', default='grade', help='target attribute: grade, type, top_label (default: grade)')
    parser.add_argument('--label', default='both', type=str, help='only when target=grade; values: ta, tva, norm or both (default: both)')
    parser.add_argument('--test', type=str, help='Run id to test', default=None)

    # misc config
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mock', action='store_true', dest='mock', help='mock dataset (random noise)')
    parser.add_argument('--seed', type=int, default=42)
    parser.set_defaults(mock=False)

    config = parser.parse_args()
    config.device = torch.device(config.device)

    main(config)
