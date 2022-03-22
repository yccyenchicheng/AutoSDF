
import torch
import torch.nn as nn

from einops import rearrange
from models.networks.pvqvae_networks.modules import ResnetBlock as PVQVAEResnetBlock, AttnBlock, Normalize
from transformers import BertTokenizer, BertModel


class BERT2VQ(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained("bert-base-uncased")

        if opt.gpu_ids[0] != -1:
            self.device = f'cuda:{opt.gpu_ids[0]}'
        else:
            self.device + "cpu"
        ntoken=512
        nblocks = 2
        use_attn = False
        convt_layers = []
        in_c = 64
        self.dz = self.hz = self.wz = 8
        
        self.linear_expand = nn.Linear(768,1024)
        self.linear_to3d = nn.Linear(1024, self.hz * self.wz * self.dz)

        self.linear3d_to_conv = torch.nn.Conv3d(1, in_c, 3, 1, 1)
        
        for _ in range(nblocks):
            out_c = in_c * 2
            convt_layers.append(PVQVAEResnetBlock(in_channels=in_c, out_channels=out_c, temb_channels=0, dropout=0.1))
            if use_attn:
                convt_layers.append( AttnBlock(out_c) )
            in_c = out_c
        
        self.convt_layers = nn.Sequential(*convt_layers)

        self.convt3 = PVQVAEResnetBlock(in_channels=in_c, out_channels=in_c, temb_channels=0, dropout=0.1)
        if use_attn:
            self.attn3 = AttnBlock(in_c)
        
        self.norm_out = Normalize(in_c)
        self.conv_out = torch.nn.Conv3d(in_c, ntoken, 3, 1, 1)
    
    def forward(self,x):
        
        # Extract BERT Features
        tokenized = self.tokenizer(x,return_tensors='pt',padding=True).to(self.device)
        x = self.bertmodel(**tokenized).pooler_output
        
        # Map to 3D space
        x = self.linear_expand(x)
        x = self.linear_to3d(x).unsqueeze(1)

        x = rearrange(x, 'b c (d h w) -> b c d h w', d=8, h=8, w=8)
        x = self.linear3d_to_conv(x)
        temb=None
        x = self.convt_layers(x)

        x = self.convt3(x, temb)
        if hasattr(self, 'attn3'):
            x = self.attn3(x)

        x = self.norm_out(x)
        x = self.conv_out(x)

        return x