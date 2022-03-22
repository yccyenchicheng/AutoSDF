import os
from termcolor import colored, cprint
import torch
import utils.util as util



def create_model(opt):
    model = None

    if opt.model == 'pvqvae':
        # vqvae
        from models.pvqvae_model import PVQVAEModel
        model = PVQVAEModel()
    elif opt.model == 'rand_tf':
        # transformer
        from models.rand_tf_model import RandTransformerModel
        model = RandTransformerModel()
    elif opt.model == 'seq_tf':
        # seq-transformer
        from models.seq_tf_model import SeqTransformerModel
        model = SeqTransformerModel()
    elif opt.model == 'bert2vq':
        from models.bert2vq_model import BERT2VQModel
        model = BERT2VQModel()
    elif opt.model == 'resnet2vq':
        from models.resnet2vq_model import ResNet2VQModel
        model = ResNet2VQModel()
    elif opt.model == 'resnet2vox':
        from models.resnet2vox_model import ResNet2VoxModel
        model = ResNet2VoxModel()
    elif opt.model == 'resnet2sdf':
        from models.resnet2sdf_model import ResNet2SDFModel
        model = ResNet2SDFModel()
    elif opt.model == 'baseline_je':
        from models.baseline_je_model import LangJEModel
        model = LangJEModel()
    elif opt.model == 'img_je':
        from models.img_je_model import ImgJEModel
        model = ImgJEModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    cprint("[*] Model has been created: %s" % model.name(), 'blue')
    return model

# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        if self.isTrain:
            # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.save_dir = os.path.join(opt.logs_dir, opt.name, 'ckpt')

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.model_names = []
        self.epoch_labels = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def save_networks(self, model_names, epoch_labels):

        assert(len(model_names) == len(epoch_labels))

        for i,name in enumerate(model_names):
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                save_filename = '%s_net_%s.pth' % (epoch_labels[i], name)
                save_path = os.path.join(self.save_dir, save_filename)
                if len(self.gpu_ids) and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)


    # # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, ckpt_path):
        raise NotImplementedError
        opt = self.opt

        if not os.path.exists(ckpt_path):
            print('[*] %s does not exist. loading from latest...' % ckpt_path)
            epoch = ckpt_path.split('epoch-')[-1].split('-')[0]
            epoch_label = 'epoch-%s' % epoch
            # prevWarpN_ckpt_dir = f'%s/%s/%s/ckpt' % (opt.checkpoints_dir, opt.name, 'warp%s' % (str(l)))
            # save_file = f'%s/%s_net_%s.pth' % (prevWarpN_ckpt_dir, epoch_label, network_label)
            ckpt_latest = ckpt_path.replace(epoch_label, 'epoch-latest')
            if not os.path.exists(ckpt_latest):
                import pdb; pdb.set_trace()
            ckpt_path = ckpt_latest
        network.load_state_dict(torch.load(ckpt_path))
        print(colored(f'[*] Network loaded from: {ckpt_path}. Done.', 'blue'))


    # helper loading function that can be used by subclasses
    def load_networks(self):
        assert(len(self.model_names) == len(self.epoch_labels))
        for i,name in enumerate(self.model_names):
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                save_filename = '%s_net_%s.pth' % (self.epoch_labels[i], name)
                save_path = os.path.join(self.save_dir, save_filename)
                net.load_state_dict(torch.load(save_path))

    # define the optimizers
    def set_optimizers(self):
        G_lr, D_lr = self.opt.lr / 2, self.opt.lr * 2
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)

                ## ttur
                if 'D_' in name or 'D1_' in name or 'D2_' in name:
                    lr = D_lr
                else:
                    lr = G_lr
                setattr(self, 'optimizer_' + name, torch.optim.Adam(net.parameters(),
                                                lr=lr, betas=(self.opt.beta1, 0.999)))
                                                # lr=self.opt.lr, betas=(self.opt.beta1, 0.999)))
                self.optimizers.append(getattr(self, 'optimizer_' + name))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('[*] learning rate = %.7f' % lr)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # print network information
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.cuda(self.gpu_ids[0], non_blocking=True))


    def tnsrs2ims(self, tensor_names):
        ims = []
        for name in tensor_names:
            if isinstance(name, str):
                var = getattr(self, name)
                ims.append(util.tensor2im(var.data))
        return ims
