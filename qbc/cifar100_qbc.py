import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
from torchvision import datasets, transforms, utils
from torchvision.models import resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
from models import *

batch_size = 64
train_epochs = 100
al_epochs = 20
lr = 0.001


def load_dataset():
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR100(
        './dataset', train=True, transform=transform, download=False)
    test_set = datasets.CIFAR100(
        './dataset', train=False, transform=transform, download=False)
    return train_set, test_set


def create_dataloader():
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR100(
        './dataset', train=True, transform=transform, download=True)
    test_set = datasets.CIFAR100(
        './dataset', train=False, transform=transform, download=False)

    # split trainset into train-val set
    train_set, val_set = torch.utils.data.random_split(train_set, [
        45000, 5000])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        # if batch_idx == 0 and epoch == 0:
        #     fig = plt.figure()
        #     inputs = inputs.detach().cpu()  # convert to cpu
        #     grid = utils.make_grid(inputs)
        #     plt.imshow(grid.numpy().transpose((1, 2, 0)))
        #     plt.show()

        # print loss and accuracy
        if(batch_idx+1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)  # must divide
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()  # Set the module in evaluation mode
    val_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(output, target).item()
            # val_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device):
    model.eval()  # Set the module in evaluation mode
    test_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # record loss and acc
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    return test_loss, correct/len(test_loader.dataset)


def main():
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join('output', now)
    os.makedirs(output_path)

    # load datasets
    train_loader, val_loader, test_loader = create_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # train validate and test
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []
    for epoch in range(train_epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            model, val_loader, device, val_loss_lst, val_acc_lst)
    test(model, test_loader, device)

    # plot loss and accuracy
    fig = plt.figure('Loss and acc')
    plt.plot(range(train_epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(train_epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(train_epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(train_epochs), val_acc_lst, 'b', label='val acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    plt.savefig(os.path.join(output_path, now + '.png'))
    # plt.show()

    # save model
    torch.save(model, os.path.join(output_path, "cifar100.pth"))


def al_qbc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load('cifar100_1.pth').to(device)
    model2 = torch.load('cifar100_2.pth').to(device)
    model3 = torch.load('cifar100_3.pth').to(device)
    # model = torch.load('cifar100.pth').to(device)
    model = resnet18(num_classes=100).to(device)

    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join('output', now)
    os.makedirs(output_path)

    # load raw dataset
    train_set, test_set = load_dataset()
    inference_batch_size = 1024

    pool = None
    left_trainset = train_set
    left_indices = list(range(len(train_set)))

    # for each trail
    trails = 10
    for trail in range(trails):
        model.eval()
        model1.eval()
        model2.eval()
        model3.eval()
        ranks = []

        inference_loader = DataLoader(
            left_trainset, batch_size=inference_batch_size, shuffle=False)

        # inference
        for batch_idx, (inputs, labels) in enumerate(inference_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs3 = model3(inputs)

            outputs1 = F.softmax(outputs1, dim=1)
            # pred[0]:conf, pred[1]:index
            pred1 = outputs1.max(1, keepdim=True)
            index1 = pred1[1].view(1, -1)[0].detach().cpu().numpy().tolist()

            outputs2 = F.softmax(outputs2, dim=1)
            # pred[0]:conf, pred[1]:index
            pred2 = outputs2.max(1, keepdim=True)
            index2 = pred2[1].view(1, -1)[0].detach().cpu().numpy().tolist()

            outputs3 = F.softmax(outputs3, dim=1)
            # pred[0]:conf, pred[1]:index
            pred3 = outputs3.max(1, keepdim=True)
            index3 = pred3[1].view(1, -1)[0].detach().cpu().numpy().tolist()

            for i in range(inputs.size(0)):
                s = set()
                s.add(index1[i])
                s.add(index2[i])
                s.add(index3[i])
                ranks.append((batch_idx*inference_batch_size+i, len(s)))
        print('inference done!')
        ranks.sort(key=lambda x: x[1], reverse=True)  # [(0,3),...]
        # print(ranks)
        selected_indices = [item[0] for item in ranks[:5000]]
        print('selected indices!')

        current_selected_trainset = Subset(
            left_trainset, selected_indices)
        if not pool:
            pool = current_selected_trainset
        else:
            pool = ConcatDataset([pool, current_selected_trainset])

        left_indices = [
            i for i in list(range(len(left_indices))) if i not in selected_indices]  # diff set
        left_trainset = Subset(left_trainset, left_indices)
        print('current trainset subed!')

        # ============================retrain===============================
        train_loader = DataLoader(
            pool, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False)

        # choose optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=lr)

        # train and test
        train_loss_lst, train_acc_lst = [], []
        for epoch in range(al_epochs):
            train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                                  epoch, device, train_loss_lst, train_acc_lst)
        test_loss, test_acc = test(model, test_loader, device)

        # write log file
        with open(os.path.join(output_path, 'log.txt'), 'a') as f:
            f.write("Trail:{}, test loss:{}, test acc:{}\n".format(
                trail, test_loss, test_acc))

        # plot loss and accuracy
        fig = plt.figure('Loss and acc')
        plt.plot(range(al_epochs), train_loss_lst, 'g', label='train loss')
        plt.plot(range(al_epochs), train_acc_lst, 'r', label='train acc')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss and acc')
        plt.legend(loc="upper right")
        now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        plt.savefig(os.path.join(output_path, 'trail'+str(trail)+'.png'))
        plt.close()

        # save model
        torch.save(model, os.path.join(
            output_path, "cifar100_qbc_trail"+str(trail)+".pth"))
        # ============================retrain===============================


if __name__ == "__main__":
    # main()
    al_qbc()
