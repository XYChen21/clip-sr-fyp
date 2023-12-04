import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from basicsr.ops.upfirdn2d.upfirdn2d import upfirdn2d
# from .sr_backbone import default_init_weights


class PixelShuffleUp(nn.Module):
    """Pixel Shuffle upsample layer.

    Copied from
    https://github.com/open-mmlab/mmagic/blob/107647a8edab3186685dc88af491d9795083523b/mmagic/models/archs/upsample.py#L9

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor=2,
                 upsample_kernel=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        # self.init_weights()

    # def init_weights(self) -> None:
    #     """Initialize weights for PixelShufflePack."""
    #     default_init_weights(self, 1)

    def forward(self, x: Tensor) -> Tensor: # x.shape=(n,c,h,w)
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class InterpolateConvUp(nn.Module):
    def __init__(self, in_chan, out_chan, scale_factor=2, kernel=3) -> None:
        super().__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.scale_factor = scale_factor
        self.upsample_kernel = kernel
        self.conv = nn.Conv2d(in_chan, out_chan, kernel, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.conv(F.interpolate(x, scale_factor=self.scale_factor, mode='nearest'))
        return self.lrelu(out)


class TransposedConvUp(nn.Module):
    def __init__(self, in_chan, out_chan, factor=2, kernel=2, blur_kernel=[1,3,3,1]) -> None:
        super().__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.scale_factor = factor
        self.kernel = kernel
        # self.blur = UpsampleUpFIRDn(kernel, True, blur_kernel, factor=2)
        self.up_conv = nn.ConvTranspose2d(in_chan, out_chan, kernel, stride=2)
        # self.conv = nn.Conv2d(out_chan, out_chan, kernel, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.up_conv(x))
        # out = self.blur(out)
        # out = self.lrelu(self.conv(out))
        return out


class ToRGB(nn.Module):
    '''
    Borrowed from
    https://github.com/open-mmlab/mmagic/blob/107647a8edab3186685dc88af491d9795083523b/mmagic/models/editors/stylegan2/stylegan2_modules.py#L396

    '''

    def __init__(self, in_chan, out_chan=3, upsample=True, blur_kernel=[1, 3, 3, 1],
                 padding=None, kernel_size=1, out_fp32=True,
                 fp16_enabled=False, conv_clamp=256):
        super().__init__()

        if upsample:
            self.upsample = UpsampleUpFIRDn(kernel_size, False, blur_kernel, factor=2)
        # add support for fp16
        # self.fp16_enabled = fp16_enabled
        # self.conv_clamp = float(conv_clamp)
        self.padding = padding if padding else (kernel_size // 2)
        self.conv = nn.Conv2d(in_chan, out_chan, 1, stride=1, padding=self.padding)

        self.bias = nn.Parameter(torch.zeros(1, out_chan, 1, 1))

        # enforece the output to be fp32 (follow Tero's implementation)
        self.out_fp32 = out_fp32

    # @auto_fp16(apply_to=('x', 'style'))
    def forward(self, x, style, skip=None):
        # with autocast(enabled=self.fp16_enabled):
        out = self.conv(x, style)
        out = out + self.bias.to(x.dtype)

        # if self.fp16_enabled:
        #     out = torch.clamp(
        #         out, min=-self.conv_clamp, max=self.conv_clamp)

            # Here, Tero adopts FP16 at `skip`.
        if skip is not None:
            if hasattr(self, 'upsample'):
                skip = self.upsample(skip)
            out = out + skip
        if self.out_fp32:
            out = out.to(torch.float32)
        return out

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class UpsampleUpFIRDn(nn.Module):
    """UpFIRDn for Upsampling.

    Borrowed from
    https://github.com/open-mmlab/mmagic/blob/107647a8edab3186685dc88af491d9795083523b/mmagic/models/editors/stylegan1/stylegan1_modules.py#L191

    https://github.com/open-mmlab/mmagic/blob/107647a8edab3186685dc88af491d9795083523b/mmagic/models/editors/stylegan2/stylegan2_modules.py#L43

    This module is used in the ``to_rgb`` layers for upsampling the images,
    or rightly after upsampling operation in StyleGAN2 to build a blurry layer.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Upsampling factor. Defaults to 2.
    """

    def __init__(self, up_kernel_size, is_blur, blur_kernel=[1,3,3,1], factor=2):
        super().__init__()
        blur_kernel = make_kernel(blur_kernel)
        self.factor = factor
        self.is_blur = is_blur
        if factor > 1:
            blur_kernel = blur_kernel * (factor**2)
        self.register_buffer('blur_kernel', blur_kernel)

        p = blur_kernel.shape[0] - factor - (up_kernel_size - 1)

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        if is_blur:
            pad1 += 1

        self.pad = (pad0, pad1)

    def forward(self, x):
        if self.is_blur:
            up = 1
        else:
            up = self.factor
        out = upfirdn2d(
            x,
            self.blur_kernel.to(x.dtype),
            up=up,
            down=1,
            pad=self.pad
        )

        return out
    
# https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/editors/stylegan2
class NoiseInjection(nn.Module):
    """Noise Injection Module.

    In StyleGAN2, they adopt this module to inject spatial random noise map in
    the generators.

    Args:
        noise_weight_init (float, optional): Initialization weight for noise
            injection. Defaults to ``0.``.
        fixed_noise (bool, optional): Whether to inject a fixed noise. Defaults
        to ``False``.
    """

    def __init__(self, noise_weight_init=0., fixed_noise=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1).fill_(noise_weight_init))
        self.fixed_noise = fixed_noise

    def forward(self, image, noise=None, return_noise=False):
        """Forward Function.

        Args:
            image (Tensor): Spatial features with a shape of (N, C, H, W).
            noise (Tensor, optional): Noises from the outside.
                Defaults to None.
            return_noise (bool, optional): Whether to return noise tensor.
                Defaults to False.

        Returns:
            Tensor: Output features.
        """
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            if self.fixed_noise:
                torch.manual_seed(1024)
                noise = torch.randn(batch, 1, height, width).cuda()

        noise = noise.to(image.dtype)
        if return_noise:
            return image + self.weight.to(image.dtype) * noise, noise

        return image + self.weight.to(image.dtype) * noise

class ModulatedConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    This module implements the modulated convolution layers proposed in
    StyleGAN2. Details can be found in Analyzing and Improving the Image
    Quality of StyleGAN, CVPR2020.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=True,
            upsample=False,
            downsample=False,
            style_bias=0.,
            padding=None,  # self define padding
            eps=1e-8,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps

        self.style_modulation = nn.Linear(style_channels, in_channels)
        # set lr_mul for conv weight
        lr_mul_ = 1.
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size,
                        kernel_size).div_(lr_mul_))

        self.padding = padding if padding else (kernel_size // 2)

    def forward(self, x, style):
        n, c, h, w = x.shape
        weight = self.weight
        # process style code
        style = self.style_modulation(style).view(n, 1, c, 1, 1) + self.style_bias
        # combine weight and style
        weight = weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

        weight = weight.view(n * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.upsample:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1,
                                      2).reshape(n * c, self.out_channels,
                                                 self.kernel_size,
                                                 self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            # x = self.blur(x)
        elif self.downsample:
            # x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.reshape(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        return x

class ModulatedStyleConv(nn.Module):
    """Modulated Style Convolution.

    In this module, we integrate the modulated conv2d, noise injector and
    activation layers into together.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to ``0.``.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 upsample=False,
                 downsample=False,
                 demodulate=True,
                 style_bias=0.,
                 fixed_noise=False):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            downsample=downsample,
            style_bias=style_bias)
        self.noise_injector = NoiseInjection(fixed_noise=fixed_noise)
        self.activate = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,
                x,
                style,
                noise=None,
                add_noise=True,
                return_noise=False):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            noise (Tensor, optional): Noise for injection. Defaults to None.
            add_noise (bool, optional): Whether apply noise injection to
                feature. Defaults to True.
            return_noise (bool, optional): Whether to return noise tensors.
                Defaults to False.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)

        if add_noise:
            if return_noise:
                out, noise = self.noise_injector(
                    out, noise=noise, return_noise=return_noise)
            else:
                out = self.noise_injector(
                    out, noise=noise, return_noise=return_noise)

        out = self.activate(out)

        return out


