import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Hyperparameters
num_epochs = 6
num_classes = 10
batch_size = 100
learning_rate = 0.001

DATA_PATH = 'C:/MNISTData'
MODEL_STORE_PATH = 'C:/MNISTData/pytorch_models'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dir_true = 'D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff\mnist/trainingSet/trainingSet/'
dir_fake = 'D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff/fake_mnist/train/'
dir_fake_test = 'D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff/fake_mnist/test/'
samp = list([50, 100, 200, 350, 500, 750, 1000, 1300, 1500, 2000])


class GANDataset(Dataset):
    def __init__(self, transform, fake_per, d_size):
        self.transform = transform
        self.labels = []
        self.images = []
        self.true_count = int(d_size * (10 - fake_per) / 10)
        self.fake_count = int(d_size * fake_per / 10)
        self.d_size = d_size
        i = 0
        h = 0
        for _ in range(self.true_count):
            image = Image.open(dir_true + str(i) + '/' + os.listdir(path=(dir_true + str(i)))[h])
            img = self.transform(image.copy())
            img.resize_(1, 28, 28)
            self.images.append(img)
            self.labels.append(i)
            if i == 9:
                i = 0
                h += 1
            else:
                i += 1
        i = 0
        h = 0
        for _ in range(self.fake_count):
            image = Image.open(dir_fake + str(i) + '/' + os.listdir(path=(dir_fake + str(i)))[h])
            img = self.transform(image.copy())
            img.resize_(1, 28, 28)
            self.images.append(img)
            self.labels.append(i)
            if i == 9:
                i = 0
                h += 1
            else:
                i += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.images[idx]
        Y = self.labels[idx]
        return X, Y


class TestGANDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.labels = []
        self.images = []
        i = 0
        for folders in os.listdir(path=dir_fake_test):
            for files in os.listdir(path=(dir_fake_test + str(i))):
                image = Image.open(dir_fake_test + str(i) + '/' + files)
                img = self.transform(image.copy())
                img.resize_(1, 28, 28)
                self.images.append(img)
                self.labels.append(i)
            i += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.images[idx]
        Y = self.labels[idx]
        return X, Y


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


test_dataset = TestGANDataset(trans)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
graph_data = list([])
for data_size in samp:
    for fakeperc in range(11):
        torch.manual_seed(8)
        # MNIST dataset
        train_dataset = GANDataset(trans, fakeperc, data_size)

        # Data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        model = ConvNet()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Run the forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if True:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))
                    torch.save(model.state_dict(),
                               'D:/soft/PyCharmCommunityEdition2019.2.3/pycharmprojects/mnist stuff/' + 'fake100.ckpt')

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total > 100:
                    break;

            print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
        graph_data.append([data_size, fakeperc, (correct/total) * 100])

fig = plt.figure()

xs = np.zeros(110)
ys = np.zeros(110)
zs = np.zeros(110)

graph_data = np.array(graph_data)
colours = list(['red', 'yellow', 'blue', 'green', 'silver', 'black', 'navy', 'grey', 'magenta', 'cyan'])

for i in range(10):
    for j in range(11):
        xs[i * 11 + j] = graph_data[i * 11 + j][0]
        ys[i * 11 + j] = graph_data[i * 11 + j][1]
        zs[i * 11 + j] = graph_data[i * 11 + j][2]
x = np.zeros((10, 11))
y = np.zeros((10, 11))
z = np.zeros((10, 11))

for i in range(11):
    for j in range(10):
        x[j][i] = xs[11 * j + i]
        y[j][i] = 10 * ys[11 * j + i]
        z[j][i] = zs[11 * j + i]

fig, ax = plt.subplots(nrows=11, ncols=1, figsize=(8, 64))
for i in range(10):
    ax[i].set_title(str(samp[i]))
    ax[i].set_xlabel("Fake data, %")
    ax[i].set_ylabel("Accuracy, %")
    ax[i].minorticks_on()
    ax[i].grid(linestyle="-.")
    ax[i].plot(y[i], z[i], color=colours[i], label=str(samp[i]))

ax[10].set_title("Everyone on one")
ax[10].set_xlabel("Fake data, %")
ax[10].set_ylabel("Accuracy, %")
ax[10].minorticks_on()
ax[10].grid(linestyle="-.")
for i in range(10):
    ax[10].plot(y[i], z[i], color=colours[i], label=str(samp[i]))

ax[10].legend(loc="lower right", ncol=4, title="Number of photos")

fig2 = plt.figure()
axes = Axes3D(fig2)
axx = fig2.gca(projection='3d')

axx.set_xlabel("Number of photos")
axx.set_ylabel("Fake data, %")
axx.set_zlabel("Accuracy, %")

for i in range(10):
    axx.plot(x[i], y[i], z[i])

plt.show()

axx.view_init(30, -120)

