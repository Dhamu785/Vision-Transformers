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