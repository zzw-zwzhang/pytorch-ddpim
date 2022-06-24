import torch as th
import numpy as np

from .method import DDPM, DDIM


def get_schedule(config):
    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, config['diffusion_step'], dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], config['diffusion_step'], dtype=np.float64)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


class Schedule(object):
    def __init__(self, args, config):
        self.args = args
        device = th.device(args.device)
        betas, alphas_cump = get_schedule(config)

        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = config['diffusion_step']

    def diffusion(self, img, t):
        noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model):
        if self.args.sample_type == 'ddim':
            img_next = DDIM(img_n, t_start, t_end, model, self.alphas_cump_pre)
            return img_next
        elif self.args.sample_type == 'ddpm':
            img_next = DDPM(img_n, t_start, t_end, model, self.alphas_cump_pre)
            return img_next
        else:
            print("Your selected sample type is not supported")
