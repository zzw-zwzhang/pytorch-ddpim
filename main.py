import os
import torch as th

from options import args_and_config
from runner.schedule import Schedule
from runner.runner import Runner
from model.ddim import Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,7,8,9"


if __name__ == "__main__":
    args, config = args_and_config()

    device = th.device(args.device)
    schedule = Schedule(args, config['Schedule'])

    model = Model(args, config['Model']).to(device)

    runner = Runner(args, config, schedule, model)
    if args.sample:
        runner.sample_fid()
    else:
        runner.train()
