import os
import argparse
import yaml
import shutil
import logging
import torch.utils.tensorboard as tb


def args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='cifar10.yml',
                        help="Choose the configs file")
    parser.add_argument("--verbose", type=str, default="info",
                        help="Verbose level: info | debug | warning | critical")

    parser.add_argument("--sample", action="store_true",
                        help="Whether to produce samples from the model",)
    parser.add_argument("--sample_speed", type=int, default=50,
                        help="Control the total generation step")
    parser.add_argument("--sample_type", type=str, default="ddim",
                        help="sampling approach (ddim or ddpm)")
    parser.add_argument("--use_pretrained", action="store_true")

    parser.add_argument("--device", type=str, default='cuda',
                        help="Choose the device to use")
    parser.add_argument("--restart", action="store_true",
                        help="Restart a previous training process")

    parser.add_argument("--exp", type=str, default="exp",
                        help="Path for saving running related data")
    parser.add_argument("--doc", type=str, default="test",
                        help="A string for documentation purpose")

    args = parser.parse_args()

    args.log_path = os.path.join(args.exp, "logs", args.doc)
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    # parse configs file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)

    if not args.sample:
        if not args.restart:
            if os.path.exists(args.log_path):
                shutil.rmtree(args.log_path)
                shutil.rmtree(tb_path)
                os.makedirs(args.log_path)
                os.makedirs(tb_path)
            else:
                if not os.path.exists(args.log_path):
                    os.makedirs(args.log_path)
                if not os.path.exists(tb_path):
                    os.makedirs(tb_path)

            with open(os.path.join(args.log_path, "configs.yml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        args.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.doc
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)

    return args, config
