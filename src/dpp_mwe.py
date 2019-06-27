import sys


import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms


# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Distributed environment parameters
dist_backend = 'nccl'
local_rank = int(sys.argv[1])
rank = int(sys.argv[2])

# Initialize the distributed environment
dist.init_process_group(dist_backend, init_method='env://', rank=rank)

dp_device_ids = [local_rank]
torch.cuda.set_device(local_rank)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())


train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=0,
                                           pin_memory=True,
                                           sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=True)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from datetime import datetime
start = datetime.now()

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print("Training complete in: " + str(datetime.now() - start))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# torch.save(model.state_dict(), 'model.ckpt')
