import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import Shuttlecock_Trajectory_Dataset
from dataset_ball import UniBall_Dataset
from test import eval_tracknet
from utils.general import ResumeArgumentParser, get_model, to_img_format, get_model_videomamba, remove_ddp_prefix, merge_args
from utils.metric import WBCELoss
from utils.visualize import plot_heatmap_pred_sample, plot_traj_pred_sample, write_to_tb
from utils.distribution import init_distributed_mode


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TrackNet', choices=['TrackNet', 'VideoMamba'], help='model type')
    parser.add_argument('--seq_len', type=int, default=8, help='sequence length of input')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size of training')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD', 'Adadelta'], help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', choices=['', 'StepLR'], help='learning rate scheduler')
    parser.add_argument('--bg_mode', type=str, default='', choices=['', 'subtract', 'subtract_concat', 'concat'], help='background mode')
    parser.add_argument('--alpha', type=float, default=-1, help='alpha of sample mixup, -1 means no mixup')
    parser.add_argument('--frame_alpha', type=float, default=-1, help='alpha of frame mixup, -1 means no mixup')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='ratio of random mask during training InpaintNet')
    parser.add_argument('--tolerance', type=float, default=4, help='difference tolerance of center distance between prediction and ground truth in input size')
    parser.add_argument('--resume_training', action='store_true', default=False, help='resume training from experiment directory')
    parser.add_argument('--seed', type=int, default=13, help='random seed')
    parser.add_argument('--save_dir', type=str, default='exp', help='directory to save the checkpoints and prediction result')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    parser.add_argument('--data_dir', type=str, default='datasets/TrackNetV2', help='directory of dataset')
    parser.add_argument('--data_type', type=str, default='TrackNet', choices=['TrackNet', 'UniBall'], help='dataset type')
    parser.add_argument('--last_only', action='store_true', default=False, help='only predict last frame result')
    parser.add_argument('--vis_step', type=int, default=100, help='visualize step')
    # VideoMamba args
    parser.add_argument('--patch_size', type=int, default=8, help='patch size')
    parser.add_argument('--d_shallow', type=int, default=3, help='image shallow embedding dim')
    parser.add_argument('--d_model', type=int, default=8, help='patch embedding dim')
    parser.add_argument('--depth', type=int, default=2, help='number of mamba blocks')
    # dataset args
    parser.add_argument('--heatmap_mode', type=str, default='gaussian', choices=['hard', 'gaussian'], help='heatmap type')
    parser.add_argument('--sigma', type=float, default=2.5, help='heatmap radius')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--port', type=int, default=29500, help='DDP port')

    args = parser.parse_args()
    
    if args.data_type == 'UniBall':
        args.last_only = True
    
    return args

def main():
    args = get_args()
    args = init_distributed_mode(args)
    param_dict = vars(args)
    num_workers = args.batch_size if args.batch_size <= 16 else 16
    
    # Load checkpoint
    print(f'Load checkpoint from {args.model_name}_best.pt...')
    assert os.path.exists(os.path.join(args.save_dir, f'{args.model_name}_best.pt')), f'No checkpoint found in {args.save_dir}'
    ckpt = torch.load(os.path.join(args.save_dir, f'{args.model_name}_best.pt'))
    param_dict = ckpt['param_dict']
    ckpt['param_dict']['resume_training'] = args.resume_training
    ckpt['param_dict']['epochs'] = args.epochs
    ckpt['param_dict']['verbose'] = args.verbose
    ckpt['param_dict']['save_dir'] = args.save_dir
    ckpt['param_dict']['data_dir'] = args.data_dir
    args = merge_args(args, ckpt['param_dict'])

    print(f'Parameters: {param_dict}')
    print(f'Load dataset...')
    data_mode = 'heatmap'
    if args.data_type == 'TrackNet':
        val_dataset = Shuttlecock_Trajectory_Dataset(split='val', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug)
        test_dataset = Shuttlecock_Trajectory_Dataset(split='test', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug)
    elif args.data_type == 'UniBall':
        val_dataset = UniBall_Dataset(split='val', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug, heatmap_mode=args.heatmap_mode, SIGMA=args.sigma)
        test_dataset = UniBall_Dataset(split='test', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug, heatmap_mode=args.heatmap_mode, SIGMA=args.sigma)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    print(f'Create {args.model_name}...')
    if args.model_name == 'VideoMamba':
        mamba_args = dict(
            patch_size=args.patch_size,
            d_shallow=args.d_shallow,
            d_model=args.d_model,
            depth=args.depth,
            seq_len=args.seq_len+1,
        )
        model = get_model_videomamba(mamba_args)
    else:
        model = get_model(args.model_name, args.seq_len, args.bg_mode)
    
    model = model.to(args.gpu)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params:', n_parameters)
    model.load_state_dict(remove_ddp_prefix(ckpt['model']))

    print(f'Start testing...')
        
    val_loss, val_res = eval_tracknet(model, val_loader, param_dict)
    print(f'Validation loss: {val_loss:.4f}')
    print(f'Validation result: ')
    header = '\t'.join([])
    result = '\t'.join()
    print()
    for k, v in val_res.items():
        print(f'{k}: {v:.4f}')

    test_loss, test_res = eval_tracknet(model, test_loader, param_dict)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test result: ')
    for k, v in test_res.items():
        print(f'{k}: {v:.4f}')

    print('Done......')

if __name__ == '__main__':
    main()