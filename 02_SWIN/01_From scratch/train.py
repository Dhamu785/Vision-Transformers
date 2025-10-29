import os
import shutil

from numpy.lib.function_base import iterable
from tqdm import tqdm
import torch as t
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class train:
    def __init__(self, train_data: DataLoader, test_data: DataLoader, epochs: int, loss: t.nn.Module, 
                    optimizer: Optimizer, device: str) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.scaler = t.GradScaler(device=device)

    def start(self, model: t.nn.Module) -> t.Tensor:
        model.to(device=self.device)

        train_loss = []
        test_loss = []
        train_len = len(self.train_data)
        test_len = len(self.test_data)

        for epoch in range(1, self.epochs+1):
            bar = tqdm(iterable=range(train_len), desc='Batch processing', unit='Batchs', colour='GREEN')
            model.train()
            epoch_train_loss = 0
            for x, lbl in self.train_data:
                x = x.to(self.device)
                lbl = lbl.to(self.device)

                with t.autocast(device_type=self.device):
                    pred = model(x)
                    ls = self.loss(pred, lbl)
                self.optimizer.zero_grad()
                self.scaler.scale(ls).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_train_loss += ls.item()

                bar.set_postfix(loss= f"{ls.item():.4f}")
                bar.update(1)
            bar.close()
            train_loss.append(epoch_train_loss/train_len)

            model.eval()
            epoch_test_loss = 0
            bar = tqdm(iterable=range(test_len), desc='Test batch processing', unit='Batchs', colour='RED')
            for x, lbl in self.test_data:
                x = x.to(self.device)
                lbl = lbl.to(self.device)

                with t.inference_mode():
                    pred = model(x)
                    ls = self.loss(pred, lbl)
                epoch_test_loss += ls.item()
                
                bar.set_postfix(loss=f"{epoch_test_loss:.4f}")
                bar.update(1)
            bar.close()
            test_loss.append(epoch_test_loss/test_len)

            sav_dir = os.path.join(os.getcwd(), 'runs')
            if os.path.exists(sav_dir):
                shutil.rmtree(sav_dir)
            os.mkdir(sav_dir)
            t.save(model.state_dict(), f"{sav_dir}/mdl-{epoch}.pt")
            print(f"{epoch}/{self.epochs} | train loss = {train_loss[-1]:.4f} | test loss = {test_loss[-1]:.4f}")

        return train_loss, test_loss