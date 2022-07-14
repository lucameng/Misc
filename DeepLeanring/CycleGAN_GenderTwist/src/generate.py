import torch
import matplotlib.pyplot as plt
from model import Discriminator, Generator
from train import load_data
from torchvision import transforms
import os
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
  
def generate(device='cuda'):
    gen_AB = Generator().eval()
    gen_BA = Generator().eval()
    gen_AB.load_state_dict(torch.load('img/last_gen_AB.pt'))
    gen_BA.load_state_dict(torch.load('img/last_gen_BA.pt'))
    file_path1 = './dataset/male/'
    file_path2 = './dataset/female/'
    data_A, data_B = load_data(file_path1, file_path2, 16)

    for _, (real_A, real_B) in enumerate(zip(data_A, data_B)):
        fake_A = gen_BA(real_B)[1].detach().numpy()
        fake_B = gen_AB(real_A)[1].detach().numpy()
        gen2 = fake_A.transpose(1,2,0)
        gen1 = fake_B.transpose(1,2,0)
        img1 = real_A[1].detach().numpy().transpose(1,2,0)
        img2 = real_B[1].detach().numpy().transpose(1,2,0)
        show_res(img1, img2, gen1, gen2)
        break

def show_res(img1, img2, gen1, gen2):
    img1, img2 = img1 / 2 + 0.5, img2 / 2 + 0.5
    gen1, gen2 = gen1 / 2 + 0.5, gen2 / 2 + 0.5
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('original')
    plt.subplot(2,2,2)
    plt.imshow(gen1)
    plt.title('generate')
    plt.subplot(2,2,3)
    plt.imshow(img2)
    plt.subplot(2,2,4)
    plt.imshow(gen2)
    plt.savefig('./img/result.png')

def test():
    with torch.no_grad():
        gen_AB = Generator().eval()
        gen_BA = Generator().eval()
        gen_AB.load_state_dict(torch.load('img/last_gen_AB.pt'))
        gen_BA.load_state_dict(torch.load('img/last_gen_BA.pt'))
        img1 = transform(Image.open('./img/test/tom.jpeg'))
        img2 = transform(Image.open('./img/test/taylor.jpeg'))
        gen1 = gen_AB(img1.unsqueeze(dim=0)).squeeze(dim=0).numpy().transpose(1,2,0)
        gen2 = gen_BA(img2.unsqueeze(dim=0)).squeeze(dim=0).numpy().transpose(1,2,0)
        img1 = img1.numpy().transpose(1,2,0)
        img2 = img2.numpy().transpose(1,2,0)
        show_res(img1, img2, gen1 ,gen2)

if __name__ == '__main__':
    generate()
    # test()
    
