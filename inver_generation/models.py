import torch
from torch import nn as nn
from torch.nn import functional as F
from .layers import GenInitialBlock, GenGeneralConvBlock, DisGeneralConvBlock, DisFinalBlock


class ProGenerator(nn.Module):
    """ Generator of the GAN network """

    def __init__(self, latent_size=512):
        """
        constructor for the Generator class
        :param latent_size: size of the latent manifold
        """

        super(ProGenerator, self).__init__()

        # state of the generator:
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = GenInitialBlock(self.latent_size)

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList([GenGeneralConvBlock(512, 256),
                                     GenGeneralConvBlock(256, 128),
                                     GenGeneralConvBlock(128, 64),
                                     GenGeneralConvBlock(64, 32),
                                     ])

        # create the ToRGB layers for various outputs:
        self.toRGB = nn.Conv2d(32, 3, (1, 1), bias=True)

    def forward(self, x):
        y = self.initial_block(x)

        for block in self.layers:
            y = block(y)

        out = self.toRGB(y)

        return out


class ProDiscriminator(nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, feature_size=512):
        """
        constructor for the class
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        """

        super(ProDiscriminator, self).__init__()

        self.feature_size = feature_size

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        from torch.nn import Conv2d
        self.fromRGB = Conv2d(3, 64, (1, 1), bias=True)

        # create the remaining layers
        self.layers = nn.ModuleList([DisGeneralConvBlock(64, 128),
                                     DisGeneralConvBlock(128, 256),
                                     DisGeneralConvBlock(256, 512),
                                     DisGeneralConvBlock(512, self.feature_size),
                                     ])
        self.final_layer = nn.Linear(self.feature_size, 1, bias=True)

    def forward(self, x):
        """
        forward pass of the discriminator
        :param x: input to the network
        :return: out => raw prediction values (WGAN-GP)
        """

        out = self.fromRGB(x)

        for layer in self.layers:
            out = layer(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        out = self.final_layer(out)

        return out


class WasGenerator(nn.Module):
    def __init__(self, latent):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = latent
        # Output_dim = 3 (number of channels)
        self.input_bn = nn.BatchNorm2d(latent)

        self.main_module = nn.Sequential(
            # Z latent vector
            nn.ConvTranspose2d(in_channels=latent, out_channels=1024, kernel_size=(4, 2), stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x2)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (256x16x8)
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
        )
            # output of main module --> Image (3x64x32)

        self.output = nn.Tanh()

    def forward(self, x):
        B, latent = x.size()
        x = x.view(B, latent, 1, 1)
        x = self.input_bn(x)
        x = self.main_module(x)
        return self.output(x)


class WasDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (3x64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (3x64x32)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x32x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x16x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)

        )
            # output of main module --> State (1024x4x2)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(4, 2), stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)
