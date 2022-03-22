
import torch
import torch.nn as nn

from einops import rearrange

from models.networks.resnet import model_urls, resnet18, resnet50
from models.networks.pvqvae_networks.modules import ResnetBlock as PVQVAEResnetBlock
from models.networks.pvqvae_networks.modules import Upsample, AttnBlock, Normalize, nonlinearity


class ResNet2VQ(nn.Module):
    
    def __init__(self, opt):
        super(ResNet2VQ, self).__init__()
        
        self.resnet = resnet18(pretrained=True).to(opt.device)

        ntoken = 512

        if opt.vq_note == 'default':
            ntoken = 512 # 8 ** 3
            self.dz = self.hz = self.wz = 8
        elif opt.vq_note == '4x4x4':
            # ntoken = 4 ** 3 # rebuttal. 4cubes of our model
            self.dz = self.hz = self.wz = 4

        # 3d decon
        # self.dz = self.hz = self.wz = 8
        self.linear_to3d = nn.Linear(8 ** 2, self.dz * self.hz * self.wz)
        in_c_convt1 = self.resnet.block.expansion * 512

        # 2048 -> 512 -> 256 -> 256 -> 512
        #  512 -> 128 ->  64 -> 64 -> 512

        nblocks = 3
        use_attn = True
        convt_layers = []
        in_c = in_c_convt1
        for i in range(nblocks):
            out_c = min(in_c // 2, ntoken)
            convt_layers.append(
                PVQVAEResnetBlock(in_channels=in_c, out_channels=out_c, temb_channels=0, dropout=0.1)
            )
            if use_attn:
                convt_layers.append( AttnBlock(out_c) )
            # convt_layers.append( Upsample(out_c, True) )
            in_c = out_c

        self.convt_layers = nn.Sequential(*convt_layers)

        self.convt3 = PVQVAEResnetBlock(in_channels=in_c, out_channels=in_c, temb_channels=0, dropout=0.1)
        self.attn3 = AttnBlock(in_c)
        # self.up3 = Upsample(in_c, True)
        self.norm_out = Normalize(in_c)
        self.conv_out = torch.nn.Conv3d(in_c, ntoken, 1)
        
    def forward(self, x):
        
        # extract image features
        x = self.resnet(x)
        
        B, C, SH, SW = x.shape
        x = rearrange(x, 'b c h w -> (b c) (h w)')
        x = self.linear_to3d(x)
        x = rearrange(x, '(b c) (d h w) -> b c d h w', b=B, c=C, d=self.dz, h=self.hz, w=self.wz) # 512, 8x8x8

        temb=None
        x = self.convt_layers(x)

        x = self.convt3(x, temb)
        if hasattr(self, 'attn3'):
            x = self.attn3(x)
            
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        
        return x