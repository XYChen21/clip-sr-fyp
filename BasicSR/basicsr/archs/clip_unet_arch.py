from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from basicsr.archs.clip_arch import build_model
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.upsample import TransposedConvUp


#TODO:CLIP preprocess
# def _transform(n_px): n_px = model.visual.input_resolution = 224
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

@ARCH_REGISTRY.register()
class CLIPUNetGenerator(nn.Module):
    def __init__(self, num_out_ch=3, scale=4) -> None:
        super().__init__()
        self.scale = scale
        clip_path = '/Users/x/Documents/GitHub/clip-sr-fyp/BasicSR/experiments/pretrained_models/CLIP/RN50.pt'
        # print(clip_path)
        with open(clip_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            clip = build_model(model.state_dict()).visual ## we only need visual part?
        # if device == "cpu":
        clip.float()
        clip.requires_grad_(False)  ## build_model already return model.eval()?
        self.channels = {'down2': 64}
        for i in range(2,6):
            factor = 2**i
            self.channels['down'+str(factor)] = 64*factor

        self.clip_feature = nn.ModuleList()
        first_down = nn.Sequential(
            clip.conv1, clip.bn1, clip.relu1,
            clip.conv2, clip.bn2, clip.relu2,
            clip.conv3, clip.bn3, clip.relu3,
        )
        self.clip_feature.append(first_down) #//2
        self.clip_feature.append(nn.Sequential(clip.avgpool, clip.layer1)) #//4
        self.clip_feature.append(clip.layer2) #//8
        self.clip_feature.append(clip.layer3) #//16
        self.clip_feature.append(clip.layer4) #//32

        self.up_layers = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(5, 1, -1):
            factor = 2**i
            in_chan = self.channels['down'+str(factor)]
            out_chan = self.channels['down'+str(factor//2)]
            self.up_layers.append(TransposedConvUp(in_chan, out_chan, factor=2))
            self.fuse_convs.append(nn.Conv2d(out_chan*2, out_chan, 3, 1, 1))
            self.bn.append(nn.BatchNorm2d(out_chan))
        # upsample (from 224//2=112 to 224)
        # out_chan=64
        if self.scale == 4:
            self.up_layers.append(TransposedConvUp(out_chan, out_chan, factor=2))
        self.conv_hr = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        self.conv_last = nn.Conv2d(out_chan, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, lq):
        lq = F.interpolate(lq, scale_factor=self.scale, mode='bicubic')
        x0 = self.clip_feature[0](lq)
        x1 = self.clip_feature[1](x0)
        x2 = self.clip_feature[2](x1)
        x3 = self.clip_feature[3](x2)
        x4 = self.clip_feature[4](x3)

        x5 = self.up_layers[0](x4)
        x5 = self.fuse_convs[0](torch.cat((x3, x5), dim=1))
        x5 = self.lrelu(self.bn[0](x5))

        x6 = self.up_layers[1](x5)
        x6 = self.fuse_convs[1](torch.cat((x2, x6), dim=1))
        x6 = self.lrelu(self.bn[1](x6))

        x7 = self.up_layers[2](x6)
        x7 = self.fuse_convs[2](torch.cat((x1, x7), dim=1))
        x7 = self.lrelu(self.bn[2](x7))
        
        x8 = self.up_layers[3](x7)
        x8 = self.fuse_convs[3](torch.cat((x0, x8), dim=1))
        x8 = self.lrelu(self.bn[3](x8))

        if self.scale == 4:
            x9 = self.up_layers[4](x8)
            out = self.conv_last(self.lrelu(self.conv_hr(x9)))
        else: # if scale==2, from 56->112
            out = self.conv_last(self.lrelu(self.conv_hr(x8)))
        return out