# inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from piq.feature_extractors import InceptionV3
from PIL import Image
from tqdm import tqdm
from model import Discriminator, Generator
import wandb
import piq
from piq import ssim, FID
from fid_dataset import DataProcess
import os


if __name__ == "__main__":
    os.makedirs('models',exist_ok=True)
    os.makedirs('samples',exist_ok=True)
    os.makedirs('fake_gen',exist_ok=True)

    wandb.login(key='bbe60953ed99662c4459f461386ecd58a2f2ee3a')

    run = wandb.init(
        project="dcgan_project"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_limit = 1000
    n_iter = 80
    batch_size = 128
    latent_dim = 128

    criterion = nn.BCELoss()

    discriminator = Discriminator().to(device)
    generator = Generator(latent_dim=latent_dim).to(device)
    model = InceptionV3()

    fixed_noise = torch.randn(64, latent_dim).to(device)

    dataset = torchvision.datasets.ImageFolder(root='./data',
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    train_size = int(len(dataset) * 0.95)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, drop_last=False)

    real_label = 1.
    fake_label = 0.

    optimizerD = torch.optim.Adam(discriminator.parameters(), lr = 2e-4, betas = (0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr = 2e-4, betas = (0.5, 0.999))

    for epoch in range(n_iter):
        generator.train()
        # Training epoch
        for i, batch in tqdm(enumerate(train_dataloader), desc='Training epoch'):

            discriminator.zero_grad()
            data = batch[0].to(device)
            label = torch.full((batch_size,), real_label, dtype=torch.float).to(device)

            output = discriminator(data).flatten()
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, latent_dim).to(device)
            fake = generator(noise)
            label = torch.full((batch_size,), fake_label, dtype=torch.float).to(device)
            output = discriminator(fake.detach()).flatten()
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            generator.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float).to(device)
            output = discriminator(fake).flatten()
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            wandb.log({"Loss D": errD.item()})
            wandb.log({"Loss G": errG.item()})
            wandb.log({"D(x)": D_x})
            wandb.log({"D(G(z1))": D_G_z1})
            wandb.log({"D(G(z2))": D_G_z2})

        # Validation epoch
        generator.eval()
        ssim_metric, fid_metric = 0., 0.

        # compute ssim
        for i, batch in tqdm(enumerate(val_dataloader), desc='Validation epoch'):
            data = batch[0].to(device)
            noise = torch.randn(data.shape[0], latent_dim).to(device)
            with torch.no_grad():
                generated_img = generator(noise)
                for j in range(len(data)):
                    cur_img = transforms.functional.to_pil_image(generated_img[i].cpu())
                    cur_img = cur_img.save('./fake_gen/gen_{}.png'.format(j))

            ssim_metric += ssim((data + 1) * 0.5, (generated_img + 1) * 0.5, data_range=1.) * data.shape[0]     

        ssim_metric /= len(val_set)
        wandb.log({"SSIM": ssim_metric})

        # compute FID
        set_real = DataProcess('./data', limit=gen_limit)
        set_fake = DataProcess('./fake_gen', limit=gen_limit)

        loader_1 = torch.utils.data.DataLoader(set_real, batch_size=1, shuffle=False)
        loader_2 = torch.utils.data.DataLoader(set_fake, batch_size=1, shuffle=False)

        fid_metric = piq.FID()
        feat_1 = fid_metric.compute_feats(loader_1, model)
        feat_2 = fid_metric.compute_feats(loader_2, model)
        fid = fid_metric.compute_metric(feat_1, feat_2)

        wandb.log({"FID": fid})

        # fixed noise generation to check progress
        filename = 'samples/fake_samples_epoch_%03d.png' % (epoch)
        with torch.no_grad():
            fake = generator(fixed_noise)
            torchvision.utils.save_image(fake[0:64,:,:,:].cpu(), filename, nrow=8)

        wandb.log({"Image progress": wandb.Image(filename)})    

        torch.save(generator.state_dict(), 'models/generator_epoch_%d.pth' % (epoch))
        torch.save(discriminator.state_dict(), 'models/discriminator_epoch_%d.pth' % (epoch))
