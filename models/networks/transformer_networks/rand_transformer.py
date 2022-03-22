# Code for Transformer module
import math
import torch
import torch.nn as nn

from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

from einops import rearrange, repeat

from .pos_embedding import PEPixelTransformer

class RandTransformer(nn.Module):
    def __init__(self, tf_conf, vq_conf=None):
        """init method"""
        super().__init__()

        # vqvae related params
        if vq_conf is not None:
            ntokens_vqvae = vq_conf.model.params.n_embed
            embed_dim_vqvae = vq_conf.model.params.embed_dim
        else:
            ntokens_vqvae = tf_conf.model.params.ntokens
            embed_dim_vqvae = tf_conf.model.params.embed_dim

        # pe
        pe_conf = tf_conf.pe
        pos_embed_dim = pe_conf.pos_embed_dim

        # tf
        mparam = tf_conf.model.params
        ntokens = mparam.ntokens
        d_tf = mparam.embed_dim
        nhead = mparam.nhead
        num_encoder_layers = mparam.nlayers_enc
        dim_feedforward = mparam.d_hid
        dropout = mparam.dropout
        self.ntokens_vqvae = ntokens_vqvae

        # Use the codebook embedding dim. Weights will be replaced by the learned codebook from vqvae.
        self.embedding_start = nn.Embedding(1, embed_dim_vqvae)
        self.embedding_encoder = nn.Embedding(ntokens_vqvae, embed_dim_vqvae)

        # position embedding
        self.pos_embedding = PEPixelTransformer(pe_conf=pe_conf)
        self.fuse_linear = nn.Linear(embed_dim_vqvae+pos_embed_dim+pos_embed_dim, d_tf)

        # transformer
        encoder_layer = TransformerEncoderLayer(d_tf, nhead, dim_feedforward, dropout, activation='relu')
        encoder_norm = LayerNorm(d_tf)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.dec_linear = nn.Linear(d_tf, ntokens_vqvae)

        self.d_tf = d_tf

        self._init_weights()

    def _init_weights(self) -> None:
        """initialize the weights of params."""

        _init_range = 0.1

        self.embedding_start.weight.data.uniform_(-1.0 / self.ntokens_vqvae, 1.0 / self.ntokens_vqvae)
        self.embedding_encoder.weight.data.uniform_(-1.0 / self.ntokens_vqvae, 1.0 / self.ntokens_vqvae)

        self.fuse_linear.bias.data.normal_(0, 0.02)
        self.fuse_linear.weight.data.normal_(0, 0.02)

        self.dec_linear.bias.data.normal_(0, 0.02)
        self.dec_linear.weight.data.normal_(0, 0.02)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

    def generate_square_id_mask(self, sz, device):
        mask = torch.eye(sz)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def forward_transformer(self, src, src_mask=None):
        output = self.encoder(src, mask=src_mask)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output

    def forward(self, inp, inp_posn, tgt_posn):
        """ Here we will have the full sequence of inp """
        device = inp.get_device()
        seq_len, bs = inp.shape[:2]
        tgt_len = tgt_posn.shape[0]

        # token embedding
        sos = inp[:1, :]
        inp_tokens = inp[1:, :]
        inp_val = torch.cat([self.embedding_start(sos), self.embedding_encoder(inp_tokens)], dim=0) * math.sqrt(self.d_tf)
        inp_posn = repeat(self.pos_embedding(inp_posn), 't pos_d -> t bs pos_d', bs=bs)
        tgt_posn = repeat(self.pos_embedding(tgt_posn), 't pos_d -> t bs pos_d', bs=bs)

        inp = torch.cat([inp_val, inp_posn, tgt_posn], dim=-1)

        # fusion
        inp = rearrange(inp, 't bs d -> (t bs) d')
        inp = rearrange(self.fuse_linear(inp), '(t bs) d -> t bs d', t=seq_len, bs=bs)

        src_mask = self.generate_square_subsequent_mask(seq_len, device)

        outp = self.forward_transformer(inp, src_mask=src_mask)
        
        outp = self.dec_linear(outp)

        return outp