import os
import glob
import tqdm
import logging
import torch as th
import torch.optim as optimi
import torch.utils.data as data
import torchvision.utils as tvu
from tqdm.auto import tqdm

from dataset import get_dataset, data_transform, inverse_data_transform
from model.ema import EMAHelper


def get_optim(params, config):
    if config['optimizer'] == 'adam':
        optim = optimi.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'],
                            betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                            eps=config['eps'])
    elif config['optimizer'] == 'sgd':
        optim = optimi.SGD(params, lr=config['lr'], momentum=0.9)
    else:
        optim = None

    return optim


class Runner(object):
    def __init__(self, args, config, schedule, model):
        self.args = args
        self.config = config
        self.diffusion_step = config['Schedule']['diffusion_step']
        self.sample_speed = args.sample_speed
        self.device = th.device(args.device)

        self.schedule = schedule
        self.model = model

    def train(self):
        schedule = self.schedule
        model = self.model
        model = th.nn.DataParallel(model)

        optim = get_optim(model.parameters(), self.config['Optim'])

        config = self.config['Dataset']
        dataset, test_dataset = get_dataset(self.args, config)
        train_loader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                       num_workers=config['num_workers'])

        config = self.config['Train']
        if config['ema']:
            ema = EMAHelper(mu=config['ema_rate'])
            ema.register(model)
        else:
            ema = None

        tb_logger = self.args.tb_logger
        epoch, step = 0, 0

        if self.args.restart:
            train_state = th.load(os.path.join(self.args.log_path, 'model.ckpt'), map_location=self.device)
            model.load_state_dict(train_state[0])
            optim.load_state_dict(train_state[1])
            epoch, step = train_state[2:4]
            if ema is not None:
                ema_state = th.load(os.path.join(self.args.log_path, 'ema.ckpt'), map_location=self.device)
                ema.load_state_dict(ema_state)

        for epoch in range(epoch, config['epoch']):

            for i, (img, y) in enumerate(train_loader):
                n = img.shape[0]
                model.train()
                step += 1

                t = th.randint(low=0, high=self.diffusion_step, size=(n // 2 + 1,))
                t = th.cat([t, self.diffusion_step - t - 1], dim=0)[:n].to(self.device)
                img = img.to(self.device)
                img = data_transform(self.config['Dataset'], img)

                img_n, noise = schedule.diffusion(img, t)
                noise_p = model(img_n, t)

                if config['loss_type'] == 'linear':
                    loss = (noise_p - noise).abs().sum(dim=(1, 2, 3)).mean(dim=0)
                elif config['loss_type'] == 'square':
                    loss = (noise_p - noise).square().sum(dim=(1, 2, 3)).mean(dim=0)
                else:
                    loss = None

                optim.zero_grad()
                loss.backward()
                try:
                    th.nn.utils.clip_grad_norm_(model.parameters(), self.config['Optim']['grad_clip'])
                except Exception:
                    pass
                optim.step()

                if ema is not None:
                    ema.update(model)

                if step % 10 == 0:
                    tb_logger.add_scalar('loss', loss, global_step=step)
                if step % self.config['Train']['snapshot_freq'] == 0:
                    logging.info(
                        f"step: {step}, loss: {loss.item()}"
                    )
                if step % self.config['Train']['validation_freq'] == 0:
                    config = self.config['Dataset']
                    model.eval()
                    skip = self.diffusion_step // self.sample_speed
                    seq = range(0, self.diffusion_step, skip)
                    noise = th.randn(16, config['channels'], config['image_size'],
                                     config['image_size'], device=self.device)
                    img = self.sample_image(noise, seq, model)
                    img = th.clamp(img * 0.5 + 0.5, 0.0, 1.0)
                    tb_logger.add_images('sample', img, global_step=step)
                    config = self.config['Train']
                    model.train()

                if step % self.config['Train']['save_freq'] == 0:
                    train_state = [model.state_dict(), optim.state_dict(), epoch, step]
                    th.save(train_state, os.path.join(self.args.log_path, 'model.ckpt'))
                    if ema is not None:
                        th.save(ema.state_dict(), os.path.join(self.args.log_path, 'ema.ckpt'))

    def sample_fid(self):
        config = self.config['Sample']
        model = self.model

        if not self.args.use_pretrained:
            state = th.load(
                os.path.join(self.args.log_path, "model.ckpt"),
                map_location=self.device,
            )
            model = model.to(self.device)
            model = th.nn.DataParallel(model)
            model.load_state_dict(state[0], strict=True)
        else:
            if self.config['Dataset']['dataset'] == "CIFAR10":
                name = "cifar10"
            elif self.config['Dataset']['dataset'] == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = os.path.join('./temp/models/ddim_' + name + ".ckpt")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(th.load(ckpt, map_location=self.device))
            model.to(self.device)
        model.eval()

        n = config['batch_size']
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = config['total_num']
        n_rounds = (total_n_samples - img_id) // n

        config = self.config['Dataset']
        skip = self.diffusion_step // self.sample_speed
        seq = range(0, self.diffusion_step, skip)
        for _ in tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
        ):
            noise = th.randn(n, config['channels'], config['image_size'],
                             config['image_size'], device=self.device)

            img = self.sample_image(noise, seq, model)

            img = inverse_data_transform(config, img)
            for i in range(img.shape[0]):
                if img_id+i > total_n_samples:
                    break
                tvu.save_image(img[i], os.path.join(self.args.image_folder, f"{img_id+i}.png"))

            img_id += n

    def sample_image(self, noise, seq, model):
        with th.no_grad():
            imgs = [noise]
            seq_next = [-1] + list(seq[:-1])

            n = noise.shape[0]

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (th.ones(n) * i).to(self.device)
                t_next = (th.ones(n) * j).to(self.device)

                img_t = imgs[-1].to(self.device)
                img_next = self.schedule.denoising(img_t, t_next, t, model)

                imgs.append(img_next.to('cpu'))

            img = imgs[-1]

            return img
