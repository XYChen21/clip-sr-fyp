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



