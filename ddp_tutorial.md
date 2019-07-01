# Tutorial on multiprocessing in Pytorch

## Motivation
### Why multiprocessing
It's faster
Apex
Multi-node
### Pytorch docs are insufficient

## The big picture

## Minimum working example with explanations
### Without multiprocessing

First, we import everything we need. 

```python {.line-numbers}
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
```

We define a very simple convolutional model for predicting MNIST. 

```python
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
```

The `main()` function will take in some arguments and run the training function. 

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)
```

And here's the train function. 

```python
def train(gpu, args):
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
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
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
```

Finally, we want to make sure the `main()` function gets called. 

```python
if __name__ == '__main__':
    main()
```

There's definitely some extra stuff in here (the number of gpus and nodes, for example) that we don't need yet, but it's helpful to put the whole skeleton in place. 

We can run this code by opening a terminal and typing `python src/mnist.py -n 1 -g 1 -nr 0`, which will train on a single gpu on a single node. 

### With multiprocessing

To do this with multiprocessing, we need a script that will launch a process for every gpu. Each process needs to know which gpu to use, and where it ranks amongst all the processes that are running. We'll need to run the script on each node. 

Let's take a look at the changes to each function. I've fenced off the new code to make it easy to find. 

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '10.57.23.164'              #
    os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################
```

I hand-waved over the arguments in the last section, but now we actually need them. 

- `args.nodes` is the total number of nodes we're going to use. 
- `args.gpus` is the number of gpus on each node. 
- `args.nr` is the rank of the current node within all the nodes, and goes from 0 to `args.nodes` - 1. 

Now, let's go through the new changes line by line: 

Line 12: Based on the number of nodes and gpus per node, we can calculate the `world_size`, or the total number of processes to run, which is equal to the total number of gpus because we're assigning one gpu to every process. 

Line 13: This tells the multiprocessing module what IP address to look at for process 0. It needs this so that all the processes can sync up initially. 

Line 14: Likewise, this is the port to use when looking for process 0. 

Line 15: Now, instead of running the train function once, we will spawn `args.gpus` processes, each of which runs `train(i, args)`, where `i` goes from 0 to `args.gpus` - 1. Remember, we run the `main()` function on each node, so that in total there will be `args.nodes` * `args.gpus` = `args.world_size` processes. 

Instead of lines 13 and 14, I could have run `export MASTER_ADDR=10.57.23.164` and `export MASTER_PORT=8888` in the terminal. 

Next, let's look at the modifications to `train`. I'll fence the new lines again. 

```python
def train(gpu, args):
    ######################################################################
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    ######################################################################
    
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    ######################################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    ######################################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
                                               
    ######################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    ######################################################################

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
    ######################################################################
                                               shuffle=False,            #
    ######################################################################
                                               num_workers=0,
                                               pin_memory=True,
    ######################################################################
                                               sampler=train_sampler)    # 
    ######################################################################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
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
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
```

Line 3: This is the global rank of the process within all of the processes (one process per GPU). We'll use this for line 6. 

Lines 4 - 6: Initialize the process and join up with the other processes. This is "blocking," meaning that the process won't continue until all processes have joined. I'm using the `nccl` backend here because the [pytorch docs](https://pytorch.org/docs/stable/distributed.html) say it's the fastest of the available ones. The `init_method` tells the process group where to look for some settings. In this case, it's looking at environment variables for the `MASTER_ADDR` and `MASTER_PORT`, which we set within `main`. I could have set the `world_size` there as well as `WORLD_SIZE`, but I'm choosing to set it here as a keyword argument, along with the global rank of the current process. 

Line 23: Wrap the model as a [`DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) model. This reproduces the model onto the GPU for the process. 

Lines 32-36: The [`DistributedSampler`](https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html) makes sure that each process gets a different slice of the training data. 

Lines 42 and 47: Use the `DistributedSampler` instead of shuffling the usual way. 

To run this on, say, 4 nodes with 8 GPUs each, we need 4 terminals (one on each node). On node 0 (as set by line 13 in `main`): 

```python src/mnist-distributed.py -n 4 -g 8 -nr 0```

Then, on the other nodes: 

```python src/mnist-distributed.py -n 4 -g 8 -nr i```

for $i \in \{1, 2, 3\}$. 


### With Apex for mixed precision
## Acknowledgments

Many thanks to the computational team at VL56 for all your work on various parts of this, especially to Stephen Kottman, who got a MWE up while I was still trying to figure out how multiprocessing in python works, and then explained it to me.