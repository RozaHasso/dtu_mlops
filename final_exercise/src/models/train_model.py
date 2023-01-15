import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


@click.group()
def cli():
    pass

class MyDataset(Dataset):
    def __init__(self, *filepaths):
        self.imgs = np.concatenate([np.load(f)["images"] for f in filepaths[0]])
        self.labels = np.concatenate([np.load(f)["labels"] for f in filepaths[0]])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return (self.imgs[idx], self.labels[idx])
    

def mnist():
    train_files = ["../../data/raw/corruptmnist/train_0.npz", "../../data/raw/corruptmnist/train_1.npz", "../../data/raw/corruptmnist/train_2.npz",
     "../../data/raw/corruptmnist/train_3.npz", "../../data/raw/corruptmnist/train_4.npz"]
    test_file = ["../../data/raw/corruptmnist/test.npz"]

    trainloader = DataLoader(MyDataset(train_files), batch_size = 64, shuffle=True)
    testloader = DataLoader(MyDataset(test_file), batch_size = 64, shuffle=True)
    return trainloader, testloader

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    '''Takes in a learning rate, plots the training curve and save the trained model'''
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    trainloader, _ = mnist()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    epochs = 15
    losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
            losses.append(running_loss)

    plt.plot(list(range(epochs)),losses)
    plt.savefig('../../reports/figures/training_curve.png')
    
    torch.save(model.state_dict(), 'trained_model.pth')

cli.add_command(train)

if __name__ == "__main__":
    cli()