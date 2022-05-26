import os 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]
        

sample_dir = 'artificial'
os.makedirs(sample_dir,exist_ok=True)

def save_samples(index,latent_tensors,show=True):
    fake_images=generator(latent_tensors)
    fake_fname= 'generated-image-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images),os.path.join(sample_dir,fake_fname),nrow=5)
    print('Saving ',fake_fname)
    if show:
        fig,ax = plt.subplots(figsize=(5,5))
        ax.set_xticks([]);ax.set_yticks([])
        ax.imshow(make_grid(denorm(fake_images.cpu().detach()),nrow=5).permute(1,2,0))

def train_discriminator(real_images,opt_d):
    
    opt_d.zero_grad()
    
    real_preds = discriminator(real_images)
    real_targets= torch.ones(real_images.size(0),1,device=device)
    real_loss = F.binary_cross_entropy(real_preds,real_targets)
    real_score = torch.mean(real_preds).item()
    
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    fake_targets = torch.zeros(fake_images.size(0),1,device=device)
    fake_preds= discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds,fake_targets)
    fake_score=torch.mean(fake_preds).item()
    
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(),real_score,fake_score

def train_generator(opt_g):
    opt_g.zero_grad()
    
    latent = torch.randn(batch_size, latent_size, 1,1, device=device)
    fake_images = generator(latent)
    
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size,1, device=device)
    loss = F.binary_cross_entropy(preds,targets)
    
    loss.backward()
    opt_g.step()
    
    return loss.item()

def fit(epochs,lr):   
    torch.cuda.empty_cache()
    
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    opt_d = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(0.5,0.999))
    opt_g = torch.optim.Adam(generator.parameters(),lr=lr,betas=(0.5,0.999))
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dataloader):
            real_images = real_images.to(device)
            loss_d,real_score, fake_score = train_discriminator(real_images,opt_d)
            
            loss_g = train_generator(opt_g)
            
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        print('Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}'.format(
            epoch+1,epochs,loss_g,loss_d,real_score,fake_score))

        if epoch%10==9: save_samples(epoch+1,fixed_latent, show=False)
    
    return losses_g,losses_d,real_scores, fake_scores

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

    
if __name__ == '__main__':
    
    DATA_DIR = 'goblins'
    image_size = 64
    batch_size = 128
    stats = ((0.5,0.5,0.5),(0.5,0.5,0.5))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs=500
    lr = 0.0002
    latent_size = 128
    fixed_latent = torch.randn(25, latent_size, 1,1, device=device)

    train_dataset = ImageFolder(DATA_DIR,transform=tt.Compose([
        tt.Resize(image_size), tt.CenterCrop(image_size), tt.ToTensor(), tt.Normalize(*stats)]))

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    )
    generator.to(device)
    
    
    discriminator = nn.Sequential(    
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
    )
    discriminator.to(device)

    
    history = fit(epochs,lr)
    x=[i+1 for i in range(epochs)]
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, history[0], )
    axs[0, 0].set_title('Generator loss')
    axs[0, 1].plot(x, history[1], 'tab:orange')
    axs[0, 1].set_title('Discriminator loss')
    axs[1, 0].plot(x, history[2], 'tab:green')
    axs[1, 0].set_title('Real scores')
    axs[1, 1].plot(x, history[3], 'tab:red')
    axs[1, 1].set_title('Fake scores')

    for ax in axs.flat: ax.set(xlabel='x-label', ylabel='y-label')
    for ax in axs.flat: ax.label_outer()
    plt.savefig("charts.jpg")
    
    torch.save(discriminator.state_dict(), 'models\discriminator_state')
    torch.save(generator.state_dict(), 'models\generator_state')