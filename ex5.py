import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np



def generate_loaders(transforms, num_worker=10):
    transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms)
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    split = int(0.2*len(train_dataset))  # define the split size
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler, num_workers=num_worker)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler, num_workers=num_worker)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_worker)
    return train_loader, validation_loader, test_loader



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x




def train(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print("done iteration in train")
    print("Done train!")


def test(model, test_loader, loader_size):
    model.eval()
    global num_of_succ
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    accuracy = 100. * correct / loader_size
    average_loss = test_loss / loader_size
    test_loss /= loader_size
    num_of_succ = correct

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / loader_size))

    return accuracy, average_loss


def test_and_print(model, loader, loader_size, type):
    acc, loss = test(model,loader, loader_size)
    print("{} set accuracy: ({:.0f}%), Average {} set loss: {:.4f}".format(type,acc, type, loss))
    return acc, loss


def run_the_model(model, train_loader, validation_loader, test_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    global maxprecent
    global max_num_of_succ

    validation_loss_dict = {}
    training_loss_dict = {}
    validation_acc_dict = {}
    training_acc_dict = {}
    for epoch in range(1, 100 + 1):
        train(model, optimizer, train_loader)
        acc, loss = test_and_print(model, train_loader, int(0.8 * len(train_loader.dataset)), "training")
        training_loss_dict[epoch] = loss
        training_acc_dict[epoch] = acc

        acc, loss = test_and_print(model, validation_loader, int(0.2 * len(train_loader.dataset)), "validation")
        validation_loss_dict[epoch] = loss
        validation_acc_dict[epoch] = acc

        acc, loss = test_and_print(model, test_loader, len(test_loader.dataset), "test")

        if (acc > maxprecent or num_of_succ > max_num_of_succ):
            print("acc ({}) > maxprecent ({})".format(acc, maxprecent))
            print("num_of_succ ({}) > max_num_of_succ ({})".format(num_of_succ, max_num_of_succ))
            max_num_of_succ = num_of_succ
            maxprecent = acc

        print("epoch: {}".format(epoch))


def main():
    net = ConvNet()
    train_loader, validation_loader, test_loader = generate_loaders(transforms)
    run_the_model(net, train_loader, validation_loader, test_loader)


if __name__ == '__main__':
    main()













def not_tested():
    net = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_loader, validation_loader, test_loader = generate_loaders(transforms)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))