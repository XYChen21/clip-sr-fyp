import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Normalize
from basicsr.archs.clip_arch import build_model
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.upsample import TransposedConvUp

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.skip(x)
        out = self.relu(out)
        
        return out

# def make_layer(block, num, **build_opt):
#     layers = []
#     for _ in range(num):
#         layers.append(block(**build_opt))
#     return nn.Sequential(*layers)

def calculate_parameters(net):
    out = 0
    for name, param in net.named_parameters():
        if 'clip_feature' in name:
            continue
        print(name, param.numel())
        out += param.numel()
    return out

@ARCH_REGISTRY.register()
class CLIPUNetGenerator(nn.Module):
    def __init__(self, num_out_ch=3, scale=4, pretrained=True, finetune=False, num_clip_features=4) -> None:
        super().__init__()
        self.scale = scale
        self.num_clip_features = num_clip_features


        clip_path = '/Users/x/Documents/GitHub/clip-sr-fyp/BasicSR/experiments/pretrained_models/CLIP/RN50.pt'
        # clip_path = '/home/xychen/basicsr/experiments/pretrained_models/CLIP/RN50.pt'
        with open(clip_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            clip = build_model(model.state_dict(), pretrained).visual 
        clip.float()
        self.clip_transform = Compose([
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.inv_clip_transform = Compose([
            Normalize(mean = (0., 0., 0.), std = (1/0.26862954, 1/0.26130258, 1/0.27577711)),
            Normalize(mean = (-0.48145466, -0.4578275, -0.40821073), std = (1., 1., 1.)),
        ])
        if pretrained and not finetune:
            clip.requires_grad_(False)
            clip.eval()


        self.channels = {}
        self.clip_feature = nn.ModuleDict()
        for i in range(1, num_clip_features+1):
            factor = 2**i
            if factor == 2:
                self.channels[f'down{factor}'] = 64
                first_down = nn.Sequential(
                    clip.conv1, clip.bn1, clip.relu1,
                    clip.conv2, clip.bn2, clip.relu2,
                    clip.conv3, clip.bn3, clip.relu3,
                )
                self.clip_feature[f'down{factor}'] = first_down #//2
            elif factor == 4:
                self.channels[f'down{factor}'] = 64*factor
                self.clip_feature[f'down{factor}'] = nn.Sequential(clip.avgpool, clip.layer1) #//4
            elif factor == 8:
                self.channels[f'down{factor}'] = 64*factor
                self.clip_feature[f'down{factor}'] = clip.layer2 #//8
            elif factor == 16:
                self.channels[f'down{factor}'] = 64*factor
                self.clip_feature[f'down{factor}'] = clip.layer3 #//16
            elif factor == 32:
                self.channels[f'down{factor}'] = 64*factor
                self.clip_feature[f'down{factor}'] = clip.layer4 #//32


        self.up_layers = nn.ModuleDict()
        self.fuse_convs = nn.ModuleDict()
        for i in range(num_clip_features-1, 0, -1):
            factor = 2**i
            in_chan = self.channels[f'down{factor*2}']
            out_chan = self.channels[f'down{factor}']
            self.up_layers[f'to{factor}'] = TransposedConvUp(in_chan, out_chan, factor=2)
            self.fuse_convs[f'fuse{factor}'] = nn.Sequential(
                ResidualBlock(out_chan*2, out_chan*2),
                ResidualBlock(out_chan*2, out_chan)
            )
        # upsample (from 224//2=112 to 224)
        # out_chan=64

        if self.scale == 4:
            self.up_layers['to1'] = TransposedConvUp(out_chan, out_chan, factor=2)
        self.conv_hr = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        self.conv_last = nn.Conv2d(out_chan, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)


    def forward(self, lq):
        lq = F.interpolate(lq, scale_factor=self.scale, mode='bicubic')
        #TODO: normalize
        x = self.clip_transform(lq)
        clip_features = {}
        for i in range(1, self.num_clip_features+1):
            factor = 2**i
            x = self.clip_feature[f'down{factor}'](x)
            clip_features[f'down{factor}'] = x

        for i in range(self.num_clip_features-1, 0, -1):
            factor = 2**i
            x = self.up_layers[f'to{factor}'](x)
            x = self.fuse_convs[f'fuse{factor}'](
                torch.cat( (x, clip_features[f'down{factor}']), dim=1 )
            )
            x = self.lrelu(x)

        if self.scale == 4:
            x = self.up_layers['to1'](x)
            out = self.conv_last(self.lrelu(self.conv_hr(x)))
        else: # if scale==2, from 56->112
            out = self.conv_last(self.lrelu(self.conv_hr(x)))
        #TODO: inverse normalize
        out = self.inv_clip_transform(out)
        return out
    
# if __name__ == '__main__':
#     test = CLIPUNetGenerator()
#     print(calculate_parameters(test))