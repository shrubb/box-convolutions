# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from box_convolution import BoxConv2d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = BoxConv2d(1, 40, 28, 28)
        self.conv1_1x1 = nn.Conv2d(40, 40, 1, 1)

        self.fc1 = nn.Linear(7*7*40, 10)

    def forward(self, x):
        x = self.conv1_1x1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 4))

        x = self.fc1(x.view(-1, 7*7*40))
        return F.log_softmax(x, dim=1)
    
import cv2
box_video_resolution = (300, 300)
box_video = cv2.VideoWriter(
    'mnist-boxes.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, box_video_resolution)
video_background = None # to be defined in `main()`, sorry for messy code

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # log boxes to the video file
        if batch_idx % 5 == 0:
            boxes_plot = model.conv1.draw_boxes(resolution=box_video_resolution)
            # print(boxes_plot.shape, boxes_plot.dtype)
            # print(video_background.shape, video_background.dtype)
            box_video.write(cv2.addWeighted(boxes_plot, 1.0, video_background, 0.3, 0.0))

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    use_cuda = False
    batch_size = 64
    n_epochs = 10

    torch.manual_seed(666)

    device = torch.device("cuda" if use_cuda else "cpu")

    mnist_train = datasets.MNIST(
        './', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))
    mnist_test = datasets.MNIST(
        './', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # defined at the top for beautiful box visualization
    global video_background
    video_background = torch.nn.functional.pad(mnist_test[2222][0], (14,14,14,14))
    video_background = torch.nn.functional.upsample_nearest(
        video_background.unsqueeze(0), size=box_video_resolution)[0,0]
    video_background = video_background.unsqueeze(-1).repeat(1, 1, 3)
    video_background = video_background.mul(255).round().byte().numpy()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        mnist_test,  batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        
if __name__ == '__main__':
    main()
