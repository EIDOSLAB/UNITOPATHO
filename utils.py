import random
import os
import numpy as np
import torch
import wandb
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, recall_score

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def binary_accuracy(outputs, labels):
    preds = (torch.sigmoid(outputs) > 0.5).long()
    correct = preds.eq(labels.long()).sum()
    return (correct.float() / float(len(outputs))).item()

def binary_ba(outputs, labels):
    preds = (torch.sigmoid(outputs) > 0.5).long()
    return balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

def roc(outputs, labels, average='macro', multi_class='raise'):
    if average is None:
        outputs = torch.softmax(outputs, dim=1)
    else:
        outputs = torch.sigmoid(outputs)
    return {c: r for c,r in enumerate(roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy(), average=average, multi_class=multi_class))}

def binary_metrics(outputs, labels):
    return dict(
        accuracy=binary_accuracy(outputs, labels),
        ba=binary_ba(outputs, labels),
        roc=roc(outputs, labels)
    )

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds.eq(labels.long()).sum().float() / labels.shape[0]).item()

def ba(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

def class_ba(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    preds = preds.cpu().numpy()
    targets = torch.unique(labels.long()).cpu().numpy()
    labels = labels.long().cpu().numpy()

    class_ba = {}
    for target in targets:
        class_labels = (labels == target).astype(np.uint8)
        class_preds = (preds == target).astype(np.uint8)
        class_ba[int(target)] = balanced_accuracy_score(class_labels, class_preds)

    return class_ba

def recall(outputs, labels, average='binary'):
    _, preds = torch.max(outputs, dim=1)
    return {c: r for c, r in enumerate(recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average=average))}

def cm(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    print(cm)
    return cm

def metrics(outputs, labels):
    return dict(
        accuracy=accuracy(outputs, labels),
        ba=ba(outputs, labels),
        class_ba=class_ba(outputs, labels),
        recall=recall(outputs, labels, average=None),
        #roc=roc(outputs, labels, average=None, multi_class='ovo'),
        cm=wandb.Table(dataframe=pd.DataFrame(cm(outputs, labels)))
    )

def train(model, dataloader, criterion, optimizer, device, metrics, accumulation_steps=1, scaler=None, verbose=True):
    num_samples, tot_loss = 0., 0.
    all_outputs, all_labels = [], []

    model.train()
    itr = tqdm(dataloader, leave=False) if verbose else dataloader
    for step, (data, labels) in enumerate(itr):
        data, labels = data.to(device), labels.to(device)

        outputs, loss = None, None

        if scaler is None:
            with torch.enable_grad():
                outputs = model(data)
                loss = criterion(outputs, labels) / accumulation_steps
        else:
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, labels) / accumulation_steps

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        if (step+1) % accumulation_steps == 0 or step == len(dataloader)-1:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad()

        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

        batch_size = data.shape[0]
        num_samples += batch_size
        tot_loss += loss.item() * accumulation_steps * batch_size


    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    tracked_metrics = metrics(all_outputs, all_labels)
    tracked_metrics.update({'loss': tot_loss / num_samples})
    return tracked_metrics

def test(model, dataloader, criterion, device, metrics):
    num_samples, tot_loss = 0., 0.
    all_outputs, all_labels = [], []

    model.eval()
    for data, labels in tqdm(dataloader, leave=False):
        data, labels = data.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, labels)

        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

        batch_size = data.shape[0]
        num_samples += batch_size
        tot_loss += loss.item() * batch_size

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    tracked_metrics = metrics(all_outputs, all_labels)
    tracked_metrics.update({'loss': tot_loss / num_samples})
    return tracked_metrics