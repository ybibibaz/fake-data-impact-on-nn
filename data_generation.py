import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torchvision
from torchvision.utils import save_image


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


'''class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.l1 = nn.Linear(64, 256)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 784)
        self.a2 = nn.Tanh()
    def forward(self, x):
        out = self.l1(x)
        out = self.a1(x)
        out = self.l2(x)
        out = self.a1(x)
        out = self.l3(x)
        out = self.a2(x)
        return out'''

modelG = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh())
modelD = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())


for i in range(10):
    modelG.load_state_dict(
        torch.load('D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff/models/G' + str(i) + '.ckpt'))
    modelG.eval()
    modelD.load_state_dict(
        torch.load('D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff/models/D' + str(i) + '.ckpt'))
    modelD.eval()
    j = 0
    while j != 1000:
        inp = torch.randn(1, 64)
        out = modelG(inp)
        if modelD(out) > 0.5:
            out = out.reshape(out.size(0), 1, 28, 28)
            save_image(denorm(out), 'D:\soft\PyCharmCommunityEdition2019.2.3\pycharmprojects\mnist stuff/fake_mnist/test/'
                       + str(i) + '/' + str(j) + '.jpg')
            j += 1
