#very very simple MNIST dataset classification example just to get to know pytorch 

import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader


class neural_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),        #takes one channel (black and white), outputs 16
            nn.ReLU(),
            nn.MaxPool2d(2),   #makes dimentions 12x12 instead of 28x28
        )

        self.fc = nn.Sequential(
            nn.Linear(16*12*12, 100),        #takes in 16 channels
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        output = self.conv(x)
        output = torch.flatten(output, start_dim=1) 
        output = self.fc(output)
        return output


if __name__ == "__main__":
    PILToTensor = torchvision.transforms.PILToTensor()
    dataset = torchvision.datasets.MNIST(root="./mnist-dataset", train=True, download=True, transform=PILToTensor)
    batch_size = 32
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    example_batch = next(iter(dataloader))
    print(f"\nExample batch: {example_batch[0].shape},   label: {example_batch[1].shape}")


    network = neural_network()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters())

    for batch, labels in dataloader:
        batch = batch.float()
        prediction = network(batch)
        loss = (criterion(prediction, labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # check:
    example_batch[0] = example_batch[0].float()
    prediction = network(example_batch[0])
    print ("\nexpected labels:")
    print (example_batch[1])
    print ("batches from the network:")
    print (prediction.argmax(dim=1))

        