"""
    This script trains a very simple box convnet on MNIST.
    If OpenCV `videoio` is available, also outputs an animation
    of boxes' evolution to 'mnist-boxes.avi'.
    Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
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
        # The following line computes responses to 40 "generalized Haar filters"
        x = self.conv1_1x1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 4))

        x = self.fc1(x.view(-1, 7*7*40))
        return F.log_softmax(x, dim=1)

try:
    import cv2
    box_video_resolution = (300, 300)
    box_video = cv2.VideoWriter(
        'mnist-boxes.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, tuple(reversed(box_video_resolution)))
    box_video_frame_count = 0
    video_background = None # to be defined in `main()`, sorry for globals and messy code
except ImportError:
    box_video = None
    print('Couldn\'t import OpenCV. Will not log boxes to a video file')


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # log boxes to a video file
        if box_video is not None:
            global box_video_frame_count
            
            # change video background
            if box_video_frame_count % 5 == 0:
                global video_background # defined at the top for beautiful box visualization
                sample_idx = torch.randint(len(train_loader.dataset), (1,)).item()
                sample_digit = train_loader.dataset[sample_idx][0]
                video_background = torch.nn.functional.pad(sample_digit, (14,14,14,14))
                video_background = torch.nn.functional.interpolate(
                    video_background.unsqueeze(0), size=box_video_resolution, mode='nearest')[0,0]
                video_background = video_background.unsqueeze(-1).repeat(1, 1, 3)
                video_background = video_background.mul(255).round().byte().numpy()

            # log boxes to the video file
            if batch_idx % 5 == 0:
                box_importances = model.conv1_1x1.weight.detach().float().abs().max(0)[0].squeeze()
                box_importances /= box_importances.max()
                boxes_plot = model.conv1.draw_boxes(
                    resolution=box_video_resolution, weights=box_importances)
                box_video.write(cv2.addWeighted(boxes_plot, 1.0, video_background, 0.25, 0.0))
                box_video_frame_count += 1

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

        for g in optimizer.param_groups:
            g['lr'] *= 0.999

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
    use_cuda = torch.cuda.is_available()
    batch_size = 64
    n_epochs = 10

    torch.manual_seed(666)

    device = torch.device('cuda' if use_cuda else 'cpu')

    mnist_train = datasets.MNIST(
        './', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(
        './', train=False, transform=transforms.ToTensor())

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
        
if __name__ == '__main__':
    main()
