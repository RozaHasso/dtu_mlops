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
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    '''Takes in a trained model and prints an accuracy'''
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, testloader = mnist()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        
        output = model(images)

        loss = criterion(output, labels)
        running_loss += loss.item()
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
    
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))        
    print(f'Accuracy: {accuracy.item()*100}% \nLoss: {running_loss}')

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()