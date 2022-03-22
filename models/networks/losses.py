# adapt from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.lpips import LPIPS

class VQLoss(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"
                 ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
        #                                          n_layers=disc_num_layers,
        #                                          use_actnorm=use_actnorm,
        #                                          ndf=disc_ndf
        #                                          ).apply(weights_init)
        # self.discriminator_iter_start = disc_start
        # if disc_loss == "hinge":
        #     self.disc_loss = hinge_d_loss
        # elif disc_loss == "vanilla":
        #     self.disc_loss = vanilla_d_loss
        # else:
        #     raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        # print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        # self.disc_factor = disc_factor
        # self.discriminator_weight = disc_weight
        # self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx=0,
                global_step=0, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {"loss".format(split): loss.clone().detach().mean(),
               "loss_codebook".format(split): codebook_loss.detach().mean(),
               "loss_nll".format(split): nll_loss.detach().mean(),
               "loss_rec".format(split): rec_loss.detach().mean(),
               "loss_p".format(split): p_loss.detach().mean(),
            #    "{}/d_weight".format(split): d_weight.detach(),
            #    "{}/disc_factor".format(split): torch.tensor(disc_factor),
            #    "{}/g_loss".format(split): g_loss.detach().mean(),
              }
        return loss, log