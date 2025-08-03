import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch as t
import torchvision as tv

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
                    val_dataloader: t.utils.data.DataLoader, epochs: int) -> None:
        self.device = device
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.scalar = t.GradScaler(device=device)

    def calc_acc(self, predictions: t.Tensor, y: t.Tensor) -> float:
        predictions = t.argmax(t.softmax(predictions, dim=1), dim=1)
        score = (predictions == y).mean()
        return score