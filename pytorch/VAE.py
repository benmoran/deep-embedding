# http://pytorch.org/docs/master/nn.html


# - Should batch norm be learnable?
# - How to add regularization? https://discuss.pytorch.org/t/simple-l2-regularization/139/3
#   - looks like we should add losses explicitly: loss += nn.L1Loss(size_average=False)(param)

# -  torch.nn.BatchNorm3d(num_features,affine=True)
# - applies batchnorm over a 5d input that is a minibatch of 4d inputs
# - (N, C, D, H, W)

# - We can choose this to be learnable or not (affine = False)
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

CUDA = torch.cuda.is_available()

class BatchNormConvRelu(nn.Module):
    def __init__(self, chans_in, chans_out, stride=1):
        super(BatchNormConvRelu, self).__init__()
        shape = 3,3,3
        padding = 1 # so we get size=same with 3x3x3 kernel
        self.conv = nn.Conv3d(chans_in, chans_out, shape, padding=padding, stride=stride)
        self.bn = nn.BatchNorm3d(C) # affine?

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x)))


class VoxResNetLayer(nn.Module):
    def __init__(self, num_channels):
        super(VoxResNetLayer, self).__init__()
        self.bnc1 = BatchNormConvRelu(num_channels, num_channels)
        self.bnc2 = BatchNormConvRelu(num_channels, num_channels)

    def forward(self, x):
        # x is (N, C=64, D,,
        y = self.bnc1(x)
        y = self.bnc2(y)
        return x + y


class VoxResEncoder(nn.Module):

    def __init__(self, latent_size, num_vr=3, num_channels=64):
        super(VoxResEncoder, self).__init__()
        self.conv = nn.Sequential([BatchNormConvRelu(1, 32),
                                         BatchNormConvRelu(32, 32),
                                         BatchNormConvRelu(32, 64, stride=2),
                                         VoxResNetLayer(64),
                                         VoxResNetLayer(64),
                                         BatchNormConvRelu(64, 64, stride=2)])
        self.dense = nn.Linear(flat_size, latent_size)


    def forward(self, x):
        y = x.view(torch.Size([1]) + x.shape)
        y= self.conv(y)
        z = y.view(-1)
        return self.dense(z)


class MehmetEncoder2mm(nn.Module):
    def __init__(self, input_size, latent_size):
        super(MehmetEncoder2mm, self).__init__()
        self.input_size = input_size
        
        self.latent_size = latent_size
        s0, s1, s2, = 2, 2, 3
        self.last_size = tuple(i // s0 // s1 // s2 for i in input_size[-3:])
        self.flat_size = int(np.prod(self.last_size))
        self.final_c = 32
        self.conv = nn.Sequential(nn.Conv3d(1, 8, (3,3,3), padding=1),
                                  nn.MaxPool3d((s0, s0, s0)),
                                  nn.Conv3d(8, 16, (3,3,3), padding=1),
                                  nn.MaxPool3d((s1, s1, s1)),
                                  nn.Conv3d(16, self.final_c, (3,3,3), padding=1),
                                  nn.MaxPool3d((s2, s2, s2)),)

        # huh, the input shape of dense depends on the output of conv
        # which itself depends on the input to the network! how can we know it?
        self.dense = nn.Linear(self.final_c*self.flat_size, 2*latent_size) # TODO: reg on weights

    def forward(self, x):
        y = self.conv(x)
        z = self.dense(y.view(-1, self.flat_size * self.final_c))
        return z.view(-1, self.latent_size, 2)

class MehmetDecoder2mm(nn.Module):
    def __init__(self, latent_size, last_size):
        super(MehmetDecoder2mm, self).__init__()                
        self.latent_size = latent_size
        self.last_size = last_size
        self.flat_size = int(np.prod(self.last_size))
        self.dense = nn.Linear(latent_size, self.flat_size) # TODO: reg on weights
        
        self.conv = nn.Sequential(nn.Conv3d(1, 32, (3,3,3), padding=1),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=3),
                                  nn.Conv3d(32, 32, (7,7,7), padding=3),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv3d(32, 32, (5,5,5), padding=2),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv3d(32, 16, (5,5,5), padding=2),
                                  nn.ReLU(),
                                  nn.Conv3d(16, 1, (5,5,5), padding=2),
        )



    def forward(self, z):
        y = self.dense(z)
        y = self.conv(y.view(-1, 1, *self.last_size))
        return y

class MehmetEncoder4mm(nn.Module):
    def __init__(self, input_size, latent_size):
        super(MehmetEncoder4mm, self).__init__()
        self.input_size = input_size
        
        self.latent_size = latent_size
        s0, s1 = 3, 3
        self.last_size = tuple(i // s0 // s1 for i in input_size[-3:])
        self.flat_size = int(np.prod(self.last_size))
        self.final_c = 32
        self.conv = nn.Sequential(nn.Conv3d(1, 16, (3,3,3), padding=1),
                                  nn.MaxPool3d((s0, s0, s0)),
                                  nn.Conv3d(16, self.final_c, (3,3,3), padding=1),
                                  nn.MaxPool3d((s1, s1, s1)),)

        # huh, the input shape of dense depends on the output of conv
        # which itself depends on the input to the network! how can we know it?
        self.dense = nn.Linear(self.final_c*self.flat_size, 2*latent_size) # TODO: reg on weights

    def forward(self, x):
        y = self.conv(x)
        z = self.dense(y.view(-1, self.flat_size * self.final_c))
        return z.view(-1, self.latent_size, 2)

class MehmetDecoder4mm(nn.Module):
    def __init__(self, latent_size, last_size):
        super(MehmetDecoder4mm, self).__init__()                
        self.latent_size = latent_size
        self.last_size = last_size
        self.flat_size = int(np.prod(self.last_size))
        self.dense = nn.Linear(latent_size, self.flat_size) # TODO: reg on weights
        
        self.conv = nn.Sequential(nn.Conv3d(1, 32, (3,3,3), padding=1),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=3),
                                  nn.Conv3d(32, 32, (7,7,7), padding=3),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=3),
                                  nn.Conv3d(32, 32, (5,5,5), padding=2),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=1),
                                  nn.Conv3d(32, 16, (5,5,5), padding=2),
                                  nn.ReLU(),
                                  nn.Conv3d(16, 1, (5,5,5), padding=2),
        )



    def forward(self, z):
        y = self.dense(z)
        y = self.conv(y.view(-1, 1, *self.last_size))
        return y
    

class VAE(nn.Module):
    def __init__(self, enc, dec):
        super(VAE, self).__init__()

        # Construct all the layers here.
        self.encoder = enc
        self.decoder = dec

    def reparameterize(self, mu_logvar):
        mu, logvar = torch.split(mu_logvar, 1, 2)
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)
        return torch.squeeze(z, -1), mu, logvar


    def forward(self, x):
        mu_logvar = self.encoder(x)        
        z, mu, logvar = self.reparameterize(mu_logvar)
        return self.decoder(z), mu, logvar




def loss_function(recon_x, x, mu, logvar):
    # Return ELBO losses:
    # - total loglikelihood \sum_{n \in batch} log p(x|z)
    # - total KLs \sum_{n \in batch} D_KL(q(z|x_n) | p(z) )
    # ELBO = negloglik + KLD
    
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    # Let's say that the loglikelihood is just a unit-variance gaussian around recon_x
    loglik = torch.distributions.Normal(recon_x, std=1).log_prob(x).sum()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loglik, KLD

def run_training(train_loader, model, optimizer, epoch=0, log_interval=100):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loglik, kld_loss = loss_function(recon_batch, data, mu, logvar)
        loss = -loglik + kld_loss
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLogLik: {:.6f}\tKLD: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loglik.data[0] / len(data),
                kld_loss.data[0] / len(data),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))    



### MAIN
if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from depiloader import DepiDataset, Normalize, ToTensor, AddChannel

    dataset = DepiDataset("../../depi", "4mm", 
                          transform=transforms.Compose([Normalize(),
                                                        ToTensor(),
                                                        AddChannel()]))

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    eg_batch = next(iter(dataloader))[0]


    latent_size = 6
    encoder = MehmetEncoder4mm(eg_batch.shape[2:], latent_size)
    decoder = MehmetDecoder4mm(encoder.latent_size, encoder.last_size)
    model = VAE(encoder, decoder)

    if CUDA:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    

    run_training(dataloader, model, optimizer)
