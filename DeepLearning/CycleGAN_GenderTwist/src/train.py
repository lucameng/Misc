import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import itertools
from torchvision import transforms
from tqdm import tqdm
from model import Generator, Discriminator
import os
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

class MyDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.images = os.listdir(self.file_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img_path = os.path.join(self.file_path, img_path)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img

def load_data(file_path1, file_path2, batch_size):
    train_set1 = MyDataset(file_path1, transform)
    train_set2 = MyDataset(file_path2, transform)
    data_A = DataLoader(train_set1, batch_size, shuffle=True, drop_last=True)
    data_B = DataLoader(train_set2, batch_size, shuffle=True, drop_last=True)
    return data_A, data_B


def trainCycleGAN(data_A, data_B, num_epochs, lrd, lrg, device='cuda'):
    gen_AB = Generator().to(device)
    gen_BA = Generator().to(device)
    dis_A = Discriminator().to(device)
    dis_B = Discriminator().to(device)
    print('training on', device)

    # 使用预训练模型
    gen_AB.load_state_dict(torch.load('./img/last_gen_AB.pt'))
    gen_BA.load_state_dict(torch.load('./img/last_gen_BA.pt'))
    dis_A.load_state_dict(torch.load('./img/last_dis_A.pt'))
    dis_B.load_state_dict(torch.load('./img/last_dis_B.pt'))

    bceloss = nn.BCELoss()
    # bceloss = nn.MSELoss()
    l1loss = nn.L1Loss()

    gen_optimizer = torch.optim.RMSprop(params=itertools.chain(gen_AB.parameters(), gen_BA.parameters()), 
                                        lr=lrg, weight_decay=5e-5)
    dis_A_optimizer = torch.optim.RMSprop(params=dis_A.parameters(), lr=lrd, weight_decay=5e-5)
    dis_B_optimizer = torch.optim.RMSprop(params=dis_B.parameters(), lr=lrd, weight_decay=5e-5)
    for epoch in range(1, num_epochs + 1):
        D_epoch_loss = 0.0
        G_epoch_loss = 0.0
        for i, (real_A, real_B) in enumerate(tqdm(zip(data_A, data_B),desc='epoch:'+str(epoch),ncols=60,total=len(data_A))):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            true_label = torch.ones(real_A.shape[0], device=device).detach()
            fake_label = torch.zeros(real_A.shape[0], device=device).detach()

            # 训练generator
            gen_optimizer.zero_grad()
            # identity loss
            same_B = gen_AB(real_B)
            identity_B_loss = l1loss(same_B, real_B)
            same_A = gen_BA(real_A)
            identity_A_loss = l1loss(same_A, real_A)

            # GAN loss
            fake_B = gen_AB(real_A)
            dis_fake_B = dis_B(fake_B)
            gan_loss_AB = bceloss(dis_fake_B, true_label)
            fake_A = gen_BA(real_B)
            dis_fake_A = dis_A(fake_A)
            gan_loss_BA = bceloss(dis_fake_A, true_label)

            # cycle consistence loss
            recovered_A = gen_BA(fake_B)
            cycle_loss_ABA = l1loss(recovered_A, real_A)
            recovered_B = gen_AB(fake_A)
            cycle_loss_BAB = l1loss(recovered_B, real_B)

            g_loss = (identity_A_loss + identity_B_loss + gan_loss_AB + gan_loss_BA
                        + cycle_loss_ABA + cycle_loss_BAB)
            g_loss.backward()
            gen_optimizer.step()

            # 训练dis_A
            dis_A_optimizer.zero_grad()
            real_out = dis_A(real_A)
            real_loss = bceloss(real_out, true_label)
            fake_out = dis_A(fake_A.detach())
            fake_loss = bceloss(fake_out, fake_label)
            dis_A_loss = real_loss + fake_loss
            dis_A_loss.backward()
            dis_A_optimizer.step()

            # 训练dis_B
            dis_B_optimizer.zero_grad()
            real_out = dis_B(real_B)
            real_loss = bceloss(real_out, true_label)
            fake_out = dis_B(fake_B.detach())
            fake_loss = bceloss(fake_out, fake_label)
            dis_B_loss = real_loss + fake_loss
            dis_B_loss.backward()
            dis_B_optimizer.step()

            with torch.no_grad():
                D_epoch_loss += (dis_A_loss + dis_B_loss).item()
                G_epoch_loss += g_loss.item()
        with torch.no_grad():
            D_epoch_loss /= i
            G_epoch_loss /= i
            print('D_loss: %.4f, G_loss: %.4f' % (D_epoch_loss, G_epoch_loss))
            log = open("./log.txt", 'a')
            print('epoch:%d D_loss: %.4f G_loss: %.4f' 
            % (epoch, D_epoch_loss, G_epoch_loss), file=log)
            log.close()

        if epoch % 10 == 0:  
            # 每10个epoch保存一次图片.
            torchvision.utils.save_image(fake_A.data[1],'img/fake_A%d.png' % epoch, normalize=True)
            torchvision.utils.save_image(fake_B.data[1],'img/fake_B%d.png' % epoch, normalize=True)
            torchvision.utils.save_image(real_A.data[1],'img/real_A%d.png' % epoch, normalize=True)
            torchvision.utils.save_image(real_B.data[1],'img/real_B%d.png' % epoch, normalize=True)

        if (epoch % 100 == 0) and (epoch != num_epochs):
            torch.save(dis_A.state_dict(), 'img/dis_A%04d.pt' % (epoch))
            torch.save(dis_B.state_dict(), 'img/dis_B%04d.pt' % (epoch))
            torch.save(gen_AB.state_dict(), 'img/gen_AB%04d.pt' % (epoch))
            torch.save(gen_BA.state_dict(), 'img/gen_BA%04d.pt' % (epoch))
    
    torch.save(dis_A.state_dict(), 'img/last_dis_A.pt')
    torch.save(dis_B.state_dict(), 'img/last_dis_B.pt')
    torch.save(gen_AB.state_dict(), 'img/last_gen_AB.pt')
    torch.save(gen_BA.state_dict(), 'img/last_gen_BA.pt')
             

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    netG = Generator()
    netD = Discriminator()
    batch_size, num_epochs = 64, 20
    lrd, lrg = 0.000005, 0.0001
    file_path1 = './dataset_30000/male/'
    file_path2 = './dataset_30000/female/'
    data_A, data_B = load_data(file_path1, file_path2, batch_size)
    trainCycleGAN(data_A, data_B, num_epochs, lrd, lrg)



