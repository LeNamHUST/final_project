from src.evaluate import f1_score
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def train_pharse(model, train_dataloader, device, loss_criteria, optimizer):
    model.train()
    training_loss = 0.0
    out_pred = torch.FloatTensor().to(device)
    out_true = torch.FloatTensor().to(device)
    for batch, (images, labels, _) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        out_true = torch.cat((out_true, labels), 0)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_criteria(pred, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        out_pred = torch.cat((out_pred, pred), 0)
    del images, labels, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return training_loss/len(train_dataloader), np.array(f1_score(out_true, out_pred)).mean()

def evaluation_pharse(model, val_dataloader, device, loss_criteria, optimizer):
    model.eval()
    val_loss = 0.0
    out_pred = torch.FloatTensor().to(device)
    out_true = torch.FloatTensor().to(device)
    with torch.no_grad():
        for batch, (images, labels, _) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            out_true = torch.cat((out_true,  labels), 0)
            pred = model(images)
            loss = loss_criteria
            val_loss += loss(pred, labels)
            out_pred = torch.cat((out_pred, pred), 0)
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return val_loss/len(val_dataloader),  np.array(f1_score(out_true, out_pred)).mean()

def sample_data(train_data):
    labels = train_data.iloc[:, 1:].values
    N, C = labels.shape
    positive = labels.sum(axis=0)/N
    negative = 1 - positive
    sample_weights = []
    for i in range(N):
        pos_classes = np.where(labels[i] == 1)[0]
        if len(pos_classes) > 0:
            min_ratio = positive[pos_classes].min()
            weight = 1.0 / (min_ratio + 1e-6)
        else:
            weight = 1.0
        sample_weights.append(weight)
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    return sampler