import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dir = 'D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff/fake_mnist/test/'
DATA_PATH = 'C:/MNISTData'
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


class GANDataset(Dataset):
    def __init__(self, transform, dir):
      self.transform = transform
      self.labels = []
      self.images = []
      self.dir = dir
      i = 0
      for folders in os.listdir(path=dir):
        for files in os.listdir(path=(dir + str(i))):
          image = Image.open(dir + str(i) + '/' + files)
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


#test_dataset = GANDataset(trans, dir)
#test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

# Convolutional neural network (two convolutional layers)
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


model = ConvNet()
model.load_state_dict(
            torch.load('D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff\models/real100.ckpt'))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels).item()
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, (correct / total) * 100))
