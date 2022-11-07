#
# @rajp
#

# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html
# https://pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html 
import torch
import torchvision
import matplotlib.pyplot as plt 
import numpy as np 
import torchsummary 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.manual_seed(876)

"""
LeNet5: on CIFAR10 dataset
    input: 32x32x3
        op: conv2d 
            kernel: 5x5
            input channels: 3
            output chennels: 6
            stride: 1
        output shape: (32+0-5)/1 + 1  
                     floor((i+2p-k)/s) + 1
    
    feature map: 28x28x6
        op: avg pool 2d
            f=2, s=2
        output shape: (28-2)/2 + 1  
                     floor((i-k)/s) + 1
    
    feature map: 14x14x6
        op: conv2d 
            kernel: 5x5
            input channels: 6
            output chennels: 16
            stride: 1
        output shape: (14+0-5)/1 + 1  
                     floor((i+2p-k)/s) + 1
    
    feature map: 5x5x16
        op: fully connected
            input: 5x5x16
            output: 400
    
    feature map: 400
        op: linear
            input: 400
            output: 120
    
    feature map: 120
        op: linear
            input: 120
            output: 84
    
    feature map: 84
        op: linear
            input: 84
            output: 10
    
"""

class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.featureExtractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5*5*16, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )
    
    def forward(self, x):
        x = self.featureExtractor(x)
        # either add flatten here on within the single sequential block of model
        x = torch.flatten(x, 1) # flatten everything but batch dimension
        logits = self.classifier(x)
        #probabilities = torch.nn.functional.softmax(logits, dim=0)
        #label = torch.argmax(probabilities, dim=1)
        #return label
        return logits

def load_data(writer):
    # load data from torch vision and then transform to torxh tensors
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainDataSet = torchvision.datasets.CIFAR10(root="./data/cifar10/", train=True, download=True, transform=transform)
    testDataSet  = torchvision.datasets.CIFAR10(root="./data/cifar10/", train=False, download=True, transform=transform)

    # create dataloader
    trainDataLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=4, shuffle=True)
    testDataLoader  = torch.utils.data.DataLoader(testDataSet, batch_size=4, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # visualize some training data
    trainIterator = iter(trainDataLoader)
    images, labels = next(trainIterator)

    img = np.transpose(images[0].numpy(), (1,2,0))
    plt.imsave("./trainExample.png", img/2 + 0.5)
    print("[INFO] sample train data: ", labels[0], classes[labels[0]])

    # Write this to tensorboard
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("Sample training CIFAR10 image", img_grid)
    writer.flush()
    
    return (trainDataSet, trainDataLoader, testDataSet, testDataLoader, classes)

def train(model, trainDataLoader, testDataLoader, epochs, device, writer):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    best_vloss = 1_000_000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(epochs):
        runningLoss = 0.0
        for i, data in enumerate(trainDataLoader):
            inputs, labels = data
            optimizer.zero_grad()

            # forward
            outputs = model(inputs.to(device))
            #print(len(outputs), outputs)
            #print(len(labels.to(device)), labels.to(device))

            loss = criterion(outputs, labels.to(device))

            # backward
            loss.backward()
            optimizer.step()

            runningLoss += loss

            # for every mini-batch
            if  i % 1000 == 999:
                print(f'[{epoch+1}, {i + 1:5d}] loss:{runningLoss/1000:.3f}')
                runningVLoss = 0.0
                model.train(False) # disable gradient calc on validation set - IMPORTANT
                for j, vdata in enumerate(testDataLoader):
                    vinputs, vlabels = vdata
                    voutputs = model(vinputs.to(device)) 
                    vloss = criterion(voutputs, vlabels.to(device))
                    runningVLoss += vloss
                model.train(True)
                avg_loss = runningLoss / 1000.0
                avg_vloss = runningVLoss / len(testDataLoader)

                writer.add_scalars("Training vs Validation Loss", 
                                    {"Training" : avg_loss, "Validation" : avg_vloss},
                                    epoch * len(trainDataLoader) + i
                )
                writer.flush()
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    model_path = 'model_{}_epoch={}.pth'.format(timestamp, epoch)
                    torch.save(model.state_dict(), model_path)
                    # This saves weights. So when you want to load them, you need the python class. check lab6
                runningLoss = 0.0
    print("Done training ...")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model = LeNet5().to(device)
    torchsummary.summary(model, (3,32,32))
    """
        device:  cuda
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1            [-1, 6, 28, 28]             456
                AvgPool2d-2            [-1, 6, 14, 14]               0
                    Conv2d-3           [-1, 16, 10, 10]           2,416
                AvgPool2d-4             [-1, 16, 5, 5]               0
                    Linear-5                  [-1, 120]          48,120
                    Linear-6                   [-1, 84]          10,164
                    Linear-7                   [-1, 10]             850
        ================================================================
        Total params: 62,006
        Trainable params: 62,006
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.01
        Forward/backward pass size (MB): 0.06
        Params size (MB): 0.24
        Estimated Total Size (MB): 0.31
        ----------------------------------------------------------------
    """
 
    # tensorboard summary writer
    writer = SummaryWriter("./runs/experiment_2/")

    (trainDataSet, trainDataLoader, testDataSet, testDataLoader, classes) = load_data(writer)

    # Visualizing the model
    # Again, grab a single mini-batch of images
    dataiter = iter(trainDataLoader)
    images, labels = next(dataiter)
    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model, images.to(device))
    writer.flush()

    # Visualizing the dataset with embeddings - more usefulf or NLP obviously
    # Select a random subset of data and corresponding labels
    def select_n_random(data, labels, n=100):
        perm = torch.randperm(len(data))
        data = torch.tensor(data[perm][:n])
        labels = torch.tensor(torch.tensor(labels)[perm][:n])
        return data, labels
        
    # Extract a random subset of data
    images, labels = select_n_random(trainDataSet.data, trainDataSet.targets)
    print(len(images), images.shape, len(labels))

    # get the class labels for each image
    class_labels = [classes[label] for label in labels]

    # log embeddings
    print(type(images))
    features = images.reshape(100, 32*32*3)
    print(len(features), features.shape, labels.shape)

    writer.add_embedding(features,
                        metadata=class_labels,
                        label_img=images.reshape(100,3,32,32), global_step=1)
    # In there is an assertion to check the shape for I1HW or I3HW
    # so reshape the image such that channels are in 2nd position
    # hence label_img = images.reshape(100,3,32,32)

    writer.flush()
    
    # There's an error in Pytorch tensorboard
    # https://stackoverflow.com/questions/63704660/projector-tab-is-blank-in-pytorch-tensorboard
    # So adding a writer.add_graph after projector shoes the PCA of the embedding..
    # Atleast as of Oct 2022
    # Visualizing the model
    # Again, grab a single mini-batch of images
    dataiter = iter(trainDataLoader)
    images, labels = next(dataiter)
    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model, images.to(device))
    writer.flush()

    # Finally .. train
    train(model, trainDataLoader, testDataLoader, 2, device, writer)

    writer.close()
    
