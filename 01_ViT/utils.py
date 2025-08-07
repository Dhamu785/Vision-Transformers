import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch as t
import torchvision as tv

import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
from tqdm import tqdm
import os
import shutil

def get_loaders(batch_size: int):
    simple_transform = transforms.Compose([transforms.ToTensor()])
    train_data = tv.datasets.MNIST(root='./data', 
                                    train=True, transform=simple_transform, download=True)
    test_data = tv.datasets.MNIST(root='./data', 
                                    train=False, transform=simple_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

class model_utils:
    def __init__(self, device: str, optimizer: t.optim.Optimizer, 
                    loss: t.nn.modules.loss._Loss,
                    train_dataloader: t.utils.data.DataLoader, 
                    val_dataloader: t.utils.data.DataLoader, 
                    epochs: int, save_model:bool=True) -> None:
        self.device = device
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.scalar = t.GradScaler(device=device)
        self.train_len = len(train_dataloader)
        self.val_len = len(val_dataloader)
        self.save_model = save_model

    def calc_acc(self, predictions: t.Tensor, y: t.Tensor) -> t.Tensor:
        predictions = t.argmax(predictions, dim=1).to(dtype=t.long)
        score = (predictions == y).float().mean()
        return score
    
    def train_model(self, model:t.nn.Module) -> Tuple[List[float],List[float],List[float],List[float]]:
        model.to(device=self.device)
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for epoch in range(1, self.epochs+1):
            running_loss = 0
            running_acc = 0
            model.train()
            bar_train = tqdm(range(self.train_len), desc="Batch trained", unit="batchs", colour="GREEN")

            for x, y in self.train_data:
                x = x.to(self.device)
                y = y.to(self.device, dtype=t.long)
                
                # 1. gradients = 0
                self.optimizer.zero_grad()
                with t.autocast(device_type=self.device):
                    # 2. Forward pass
                    predictions = model(x)
                    # 3. Calculate the loss
                    loss : t.Tensor = self.loss(predictions, y)

                acc = self.calc_acc(predictions=predictions.detach().clone(), y=y)
                # 4. Scale the loss
                self.scalar.scale(loss).backward()
                # 5. Safely apply the gradient updates
                self.scalar.step(self.optimizer)
                # 6. Update the scale factor based on success/failure
                self.scalar.update()

                running_acc += acc.item()
                running_loss += loss.item()

                bar_train.set_postfix(loss=loss.item(), acc=acc.item())
                bar_train.update(1)
            
            bar_train.close()
            train_acc.append(running_acc/self.train_len)
            train_loss.append(running_loss/self.train_len)

            bar_val = tqdm(range(self.val_len), desc='Validation', unit='batches', colour='RED')
            model.eval()
            running_val_acc = 0
            running_val_loss = 0
            for x,y in self.val_data:
                x = x.to(self.device)
                y = y.to(self.device, dtype=t.long)

                with t.inference_mode():
                    with t.autocast(device_type=self.device):
                        predictions = model(x)
                        loss = self.loss(predictions, y)
                    acc = self.calc_acc(predictions=predictions.detach().clone(), y=y)
                    running_val_acc += acc.item()
                    running_val_loss += loss.item()
                bar_val.update(1)
            
            bar_val.close()
            val_acc.append(running_val_acc/self.val_len)
            val_loss.append(running_val_loss/self.val_len)
            sav_loc = os.path.join(os.getcwd(), 'runs')
            if self.save_model:
                if os.path.exists(sav_loc):
                    if epoch == 1:
                        shutil.rmtree(sav_loc)
                        os.mkdir(sav_loc)
                else:
                    os.mkdir(sav_loc)
                t.save(model.state_dict(), os.path.join(sav_loc, f'model-{epoch}.pt'))
        
            print(f'{epoch}/{self.epochs} | train: loss = {train_loss[-1]:.4f}, acc = {train_acc[-1]:.4f} | val: loss = {val_loss[-1]:.4f}, acc = {val_acc[-1]:.4f}')

        return train_loss, train_acc, val_loss, val_acc
    
def plot_data(data):
    x,y = data[0], data[1]
    plt.figure(figsize=(10,8))
    for i in range(1, 16):
        plt.subplot(5,5,i)
        plt.imshow(x[i][0].numpy(), cmap='gray')
        plt.title(y[i].item())
        plt.axis('off')
    plt.show()