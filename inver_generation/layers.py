# -*- coding: utf-8 -*-
# Time    : 2020/9/5 15:27
# Author  : Yichen Lu

import torch
from torch import nn as nn
from torch.nn import functional as F


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class GenInitialBlock(nn.Module):
    """ Module implementing the initial block of the input """

    def __init__(self, in_channels):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        """
        super(GenInitialBlock, self).__init__()

        self.conv_1 = nn.ConvTranspose2d(in_channels, 512, (4, 4), bias=True)
        self.conv_2 = nn.Conv2d(512, 512, (3, 5), padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # convert the tensor shape
        B, C = x.size()
        x = x.view(B, C, 1, 1)

        # perform the forward computations:
        x = self.lrelu(self.conv_1(x))
        x = self.lrelu(self.conv_2(x))

        # apply pixel norm
        x = self.pixNorm(x)

        return x


class GenGeneralConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        """

        super(GenGeneralConvBlock, self).__init__()

        self.upsampler = lambda x: F.interpolate(x, scale_factor=2)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsampler(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y


class DisGeneralConvBlock(nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        from torch.nn import AvgPool2d, LeakyReLU

        super(DisGeneralConvBlock, self).__init__()

        from torch.nn import Conv2d
        self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)

        self.downSampler = AvgPool2d(2)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        """
        from torch.nn import LeakyReLU

        super(DisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        from torch.nn import Conv2d
        self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
        # final conv layer emulates a fully connected layer
        self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
