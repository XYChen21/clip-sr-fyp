import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.transforms import Compose, Normalize
# from basicsr.archs.clip_arch import build_model
from basicsr.utils.registry import ARCH_REGISTRY
# from basicsr.utils.upsample import ModulatedStyleConv
    
class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_chan)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        if in_chan != out_chan:
            self.skip = self.skip = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 1),
                nn.BatchNorm2d(out_chan)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        out += self.skip(x)
        out = self.lrelu(out)
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ConvBlock, self).__init__()
        self.conv1 = ResidualBlock(in_chan, out_chan)
        self.conv2 = ResidualBlock(out_chan, out_chan)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DownBlock, self).__init__()
        self.down = nn.Conv2d(in_chan, in_chan, kernel_size=2, stride=2)
        self.convs = ConvBlock(in_chan, out_chan)

    def forward(self, x):
        x = self.down(x)
        x = self.convs(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        self.convs = ConvBlock(out_chan*2, out_chan)

    def forward(self, x_down, x_up):
        x_up = self.up(x_up)
        x = torch.cat((x_up, x_down), dim=1)
        x = self.convs(x)
        return x

    
def calculate_parameters(net):
    out = 0
    for name, param in net.named_parameters():
        if 'clip' in name:
            continue
        print(name, param.numel())
        out += param.numel()
    return out

    
@ARCH_REGISTRY.register()
class UNetGenerator(nn.Module):
    def __init__(self, num_out_ch=3, scale=4, pretrained=True, finetune=False, num_downsamples=4) -> None:
        super().__init__()
        self.scale = scale
        self.num_downsamples = num_downsamples

        # clip_path = '/Users/x/Documents/GitHub/clip-sr-fyp/BasicSR/experiments/pretrained_models/CLIP/RN50.pt'
        # clip_path = '/home/xychen/basicsr/experiments/pretrained_models/CLIP/RN50.pt'
        # with open(clip_path, 'rb') as opened_file:
        #     model = torch.jit.load(opened_file, map_location="cpu").eval()
        #     self.clip = build_model(model.state_dict(), pretrained).visual.float()

        # self.clip_transform = Compose([
        #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        # self.inv_clip_transform = Compose([
        #     Normalize(mean = (0., 0., 0.), std = (1/0.26862954, 1/0.26130258, 1/0.27577711)),
        #     Normalize(mean = (-0.48145466, -0.4578275, -0.40821073), std = (1., 1., 1.)),
        # ])
        # if pretrained and not finetune:
        #     self.clip.requires_grad_(False)
        #     self.clip.eval()


        self.channels = {}
        self.first = ConvBlock(in_chan=3, out_chan=64)
        self.down_layers = nn.ModuleDict()
        for i in range(1, num_downsamples+1):
            factor = 2**i
            self.channels[f'down{factor}'] = 64*factor
            self.down_layers[f'down{factor}'] = DownBlock(in_chan=64*factor//2, out_chan=64*factor)

        self.up_layers = nn.ModuleDict()
        for i in range(num_downsamples-1, 0, -1):
            factor = 2**i
            in_chan = self.channels[f'down{factor*2}']
            out_chan = self.channels[f'down{factor}']
            self.up_layers[f'up{factor}'] = UpBlock(in_chan=in_chan, out_chan=out_chan)
        
        self.last = nn.ConvTranspose2d(out_chan, out_chan, kernel_size=2, stride=2)
        self.conv_hr = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        self.conv_last = nn.Conv2d(out_chan, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)


    def forward(self, lq):
        lq = F.interpolate(lq, scale_factor=self.scale, mode='bicubic')

        x = self.first(lq)
        downsamples = {}
        for i in range(1, self.num_downsamples+1):
            factor = 2**i
            x = self.down_layers[f'down{factor}'](x)
            downsamples[f'down{factor}'] = x

        for i in range(self.num_downsamples-1, 0, -1):
            factor = 2**i
            x = self.up_layers[f'up{factor}'](downsamples[f'down{factor}'], x)

        x = self.lrelu(self.last(x))
        out = self.conv_last(self.lrelu(self.conv_hr(x)))
        out = out + lq
        return out
# 34M
# if __name__ == '__main__':
#     test = UNetGenerator()
#     print(calculate_parameters(test))