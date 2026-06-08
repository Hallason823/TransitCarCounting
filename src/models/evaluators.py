import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from .models import ResNetCounter, EfficientNetCounter
from .utils.averageMeter import AverageMeter

MODEL_NAMES = {0: 'RESNET', 1: 'EFFICIENTNET'}
CHECKPOINT_DIR = 'src/models/checkpoints'

class EvaluatorNetworks:
    """
    Trains a pretrained backbone (ResNet18 or EfficientNet-B0) for car count
    classification and evaluates it on a test set.

    model_idx:
        0 -> ResNetCounter
        1 -> EfficientNetCounter

    Best checkpoint (lowest val loss) is saved to:
        src/models/checkpoints/<MODEL_NAME>_best.pth
    """

    def __init__(self, model_idx, device, train_ds, val_ds, num_classes, optimizer_name='Adam', learning_rate=0.0001, n_epochs=30, batch_size=32, freeze_backbone=True):
        self.model_idx       = model_idx
        self.device          = device
        self.train_ds        = train_ds
        self.val_ds          = val_ds
        self.num_classes     = num_classes
        self.optimizer_name  = optimizer_name
        self.learning_rate   = learning_rate
        self.n_epochs        = n_epochs
        self.batch_size      = batch_size
        self.freeze_backbone = freeze_backbone
        self.best_val_loss   = float('inf')
        self.checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAMES.get(model_idx, 'MODEL')}_best.pth")
        self.initializeModelAndLoss()
        self.initializeOptimizer()
        self.loadDataloaders()
        self.trainNetwork()

    def initializeCheckpointDir(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def initializeModelAndLoss(self):
        if self.model_idx == 0:
            self.model = ResNetCounter(self.num_classes, self.freeze_backbone).to(self.device)
        elif self.model_idx == 1:
            self.model = EfficientNetCounter(self.num_classes, self.freeze_backbone).to(self.device)
        else:
            print('\nInitialize a valid model!\n')
            return
        self.loss_fn = nn.CrossEntropyLoss()

    def initializeOptimizer(self):
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(trainable, lr=self.learning_rate)
        else:
            print('\nInitialize a valid optimizer!\n')

    def loadDataloaders(self):
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.val_dl   = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def trainStep(self):
        self.model.train()
        train_loss = AverageMeter()
        correct, total = 0, 0
        for images, labels in self.train_dl:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item(), len(images))
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
        self.history['train_loss'].append(train_loss.avg)
        self.history['train_acc'].append(correct / total)

    def valStep(self):
        self.model.eval()
        val_loss = AverageMeter()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                val_loss.update(loss.item(), len(images))
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)
        self.history['val_loss'].append(val_loss.avg)
        self.history['val_acc'].append(correct / total)

    def saveCheckpoint(self, epoch):
        self.initializeCheckpointDir()
        torch.save({
            'epoch':       epoch,
            'model_idx':   self.model_idx,
            'num_classes': self.num_classes,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'val_loss':    self.best_val_loss,
            'history':     self.history,
        }, self.checkpoint_path)
        print(f"  >> Best model saved: {self.checkpoint_path}")

    def trainNetwork(self):
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        for epoch in range(self.n_epochs):
            self.trainStep()
            self.valStep()
            val_loss = self.history['val_loss'][epoch]
            print(f"Epoch [{epoch + 1}/{self.n_epochs}]")
            print("-" * 40)
            print(f"Train  loss: {self.history['train_loss'][epoch]:.4f}  "
                  f"acc: {self.history['train_acc'][epoch] * 100:.2f}%")
            print(f"Valid  loss: {val_loss:.4f}  "
                  f"acc: {self.history['val_acc'][epoch] * 100:.2f}%")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.saveCheckpoint(epoch + 1)
            print()

    def predictDataset(self, dataset):
        """Returns (predictions, true_labels) lists for an entire dataset."""
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for images, labels in dl:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.tolist())
        return all_preds, all_labels