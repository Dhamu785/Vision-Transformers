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