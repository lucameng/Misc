import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import random

class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(x).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(x).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*x.shape)

        return self.gamma * out + x
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, dim=256, fig_size=128):
        super().__init__()
        self.l1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, dim // 4, 7, 2, 3)),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(),)
        self.l2 = nn.Sequential(
            spectral_norm(nn.Conv2d(dim // 4, dim // 2, 3, 2, 1)),
            nn.LayerNorm([128, 32, 32]),
            nn.ReLU(),)
        self.l3 = nn.Sequential(
            spectral_norm(nn.Conv2d(dim // 2, dim, 3, 2, 1)),
            nn.LayerNorm([256, 16, 16]),
            nn.ReLU(),)
        self.attn1 = selfattention(dim)
        self.l4 = nn.Sequential(
            spectral_norm(nn.Conv2d(dim, dim * 2, 3, 2, 1)),
            nn.LayerNorm([512, 8, 8]),
            nn.ReLU(),)
        self.attn2 = selfattention(dim * 2)
        self.l5 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)),
            nn.LayerNorm([256, 16, 16]),
            nn.ReLU(),)
        self.attn3 = selfattention(dim)
        self.l6 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1)),
            nn.LayerNorm([128, 32, 32]),
            nn.ReLU(),)
        self.l7 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1)),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(),)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),)
    def forward(self, x):
        x = self.l3(self.l2(self.l1(x)))
        x = self.attn1(x)
        x = self.attn2(self.l4(x))
        x = self.attn3(self.l5(x))
        x = self.l7(self.l6(x))
        x = self.last(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
                nn.LeakyReLU(0.1)
            )
        self.l1 = conv2d(in_channels, ndf, 4, 2, 1) # [64 64 64]
        self.l2 = conv2d(ndf, ndf*2, 4, 2, 1)       # [128 32 32]
        self.l3 = conv2d(ndf*2, ndf*4, 4, 2, 1)     # [256 16 16]
        self.attn1 = selfattention(ndf*4)
        self.l4 = conv2d(ndf*4, ndf*8, 4, 2, 1)     # [512 8 8]
        self.attn2 = selfattention(ndf*8) 
        self.l5 = conv2d(ndf*8, ndf*16, 4 ,2, 1)    # [1024 4 4]
        self.l6 = nn.Conv2d(ndf*16, 1, kernel_size=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.l3(self.l2(self.l1(x)))
        x = self.attn1(x)
        x = self.l4(x)
        x = self.attn2(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.sigmoid(x).view(-1)
        return x

if __name__ == "__main__":
    inp = torch.randn(1, 3, 128, 128)
    G = Generator()
    out = G(inp)
    D = Discriminator()
    out = D(out)
    print(out.shape)
    
    # torch.save(model.state_dict(), './mymodel.pt')