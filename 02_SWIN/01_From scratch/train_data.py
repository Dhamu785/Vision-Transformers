import matplotlib.pyplot as plt
import time

import torch as t
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

st = time.time()
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

IMAGE_SIZE = 224
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.3),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
])

train_dataset = datasets.Food101(root="./data", split='train', download=False, transform=train_transform)
val_dataset = datasets.Food101(root="./data", split='test', download=False, transform=val_transform)

BATCH_SIZE = 32*4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train length = {len(train_loader)*BATCH_SIZE}, Test length = {len(val_loader)*BATCH_SIZE}")

from config import Config
from swin import swin_transformer
import train

epochs = 100
lr = 1e-4
num_classes = 211

swin_model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=num_classes).to(DEVICE)

model_path = "F:\\CBIR\\SWIN\\runs1\\mdl-100.pt"
swin_model.load_state_dict(t.load(model_path, map_location=t.device(DEVICE), weights_only=True), strict=False)

loss = t.nn.CrossEntropyLoss()
optimizer = t.optim.AdamW(swin_model.parameters(), lr=lr)

training = train.Trainer(train_loader, val_loader, epochs, loss, optimizer, DEVICE)
history = training.start(swin_model)

training_time = time.time()

x, lbl = next(iter(val_loader))
y = swin_model(x.to(DEVICE))
print(y)

lbls = {v:k for k,v in train_dataset.class_to_idx.items()}
def plot(imgs, predictions, true, lbl) -> None:
    plt.figure(figsize=(20, 20))
    for i in range(1, 26):
        plt.subplot(5, 5, i)
        plt.imshow(imgs[i-1].permute(1,2,0).cpu().numpy())
        # print(true[i-1])
        true_lbl = lbl[true[i-1].item()]
        pred_lbl = lbl[predictions[i-1].item()]
        if true_lbl == pred_lbl:
            plt.title(f"{pred_lbl}", color='green')
        else:
            plt.title(f"act: {true_lbl}\npred: {pred_lbl}", color='red')
        plt.axis('off')
    plt.savefig("Predictions.jpg")
    plt.close()

plot(imgs=x, predictions=t.argmax(y, 1), true=lbl, lbl=lbls)

plt.plot(history[0], label='train loss')
plt.plot(history[1], label='test loss')
plt.legend()
plt.savefig("Training history.jpg")
plt.close()

print(f"Model training time = {(st-training_time)/60}Min\nTotal time taken = {(st - time.time())/60}Min")