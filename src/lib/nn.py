import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import itertools

from datetime import datetime
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Colormap


def train_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    train_loss = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        
    return np.mean(train_loss)


def compute_loss(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    with torch.no_grad():
        loss_total = []
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_total.append(loss.item())

    return np.mean(loss_total)


def compute_accuracy(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    n_correct = 0
    n_total = 0
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        
        n_correct += (predictions == targets).sum()
        n_total += targets.shape[0]
    
    model.train()
    return n_correct / n_total


def get_confusion_matrix(
    loader: torch.utils.data.DataLoader, 
    model: nn.Module, 
    device: torch.device
) -> np.ndarray:
    y_test = loader.dataset.targets
    p_test = np.array([])

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        p_test = np.concatenate((p_test, predictions.cpu().numpy()))

    return confusion_matrix(y_test, p_test)


def plot_confusion_matrix(
    cm: np.ndarray, 
    classes: list[str], 
    normalize: bool = False, 
    title: str = 'Confusion matrix', 
    cmap: str | Colormap = plt.cm.Blues
):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt), horizontalalignment="center", 
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def batch_gd(
    model: nn.Module, 
    criterion: torch.nn.modules.loss._Loss, 
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader, 
    test_loader: torch.utils.data.DataLoader, 
    epochs: int,
    device: torch.device,
    stop_overfit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        time_start = datetime.now()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        model.eval()
        test_loss = compute_loss(model, criterion, test_loader, device)

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - time_start
        
        print(
            f'Epoch {it+1}/{epochs}, '
            f'Train Loss: {train_loss:.4f}, '
            f'Test Loss: {test_loss:.4f}, Duration: {dt}'
        )

        if stop_overfit and it > 0 and test_losses[it] > test_losses[it-1]:
            print('Model overfitting, early stopping...')
            break

    return train_losses[0:it+1], test_losses[0:it+1]
