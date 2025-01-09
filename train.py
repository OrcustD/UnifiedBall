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

def mixup(x, y, alpha=0.5):
    """Returns mixed inputs, pairs of targets.
    
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            alpha (float): Alpha of beta distribution

        Returns:
            x_mix (torch.Tensor): Mixed input tensor
            y_mix (torch.Tensor): Mixed target tensor
    """

    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)

    return x_mix, y_mix

def get_random_mask(mask_size, mask_ratio):
    """ Generate random mask by binomial distribution.
        1 means masked, 0 means not.
    
        Args:
            mask_size (Tuple[int, int]): Mask size (N, L)
            mask_ratio (float): Ratio of masked area

        Returns:
            mask (torch.Tensor): Random mask tensor with shape (N, L, 1)
    """

    mask = np.random.binomial(1, mask_ratio, size=mask_size)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(-1)

    return mask

def train_tracknet(model, optimizer, train_loader, param_dict, exp_name, tb_writer=None, last_only=False, curr_step=0, display_step=100, gpu=0):
    """ Train TrackNet model for one epoch.

        Args:
            model (torch.nn.Module): TrackNet model
            optimizer (torch.optim): Optimizer
            data_loader (torch.utils.data.DataLoader): Data loader
            param_dict (Dict): Parameters
                - param_dict['alpha'] (float): Alpha of sample mixup
                - param_dict['verbose'] (bool): Control whether to show progress bar
                - param_dict['bg_mode'] (str): For visualizing current prediction
                - param_dict['save_dir'] (str): For saving current prediction
        
        Returns:
            (float): Average loss
    """
    model.train()
    epoch_loss = []

    if param_dict['verbose']:
        data_prob = tqdm(train_loader)
    else:
        data_prob = train_loader
    
    display_step = min(display_step, len(train_loader))

    for step, (_, x, y, c, _) in enumerate(data_prob):
        optimizer.zero_grad()
        x, y = x.float().cuda(), y.float().cuda()

        # Sample mixup
        if param_dict['alpha'] > 0:
            x, y = mixup(x, y, param_dict['alpha'])
        
        y_pred = model(x)
        if last_only:
            y_pred = y_pred[:, -1:, :, :]
        loss = WBCELoss(y_pred, y)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if tb_writer is not None:
            tb_writer.add_scalars(f"Train_Loss/WBCE", {f'{exp_name}': loss.item()}, curr_step+step)
            tb_writer.flush()

        if param_dict['verbose'] and (step + 1) % display_step == 0:
            data_prob.set_description(f'Training')
            data_prob.set_postfix(loss=loss.item())

        # Visualize current prediction
        if (step + 1) % display_step == 0 and gpu == 0:
            x, y, y_pred = x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            c = c.numpy()
            
            # Transform inputs to image format (N, L, H , W, C)
            if param_dict['bg_mode'] == 'subtract':
                x = to_img_format(x)
            elif param_dict['bg_mode'] == 'subtract_concat':
                x = to_img_format(x, num_ch=4)
            elif param_dict['bg_mode'] == 'concat':
                x = to_img_format(x, num_ch=3)
                x = x[:, 1:, :, :, :]
            else:
                x = to_img_format(x, num_ch=3)
            
            vis_idx = np.random.randint(0, x.shape[0])
            if last_only:
                y = to_img_format(y)
                y_pred = to_img_format(y_pred[:, -1:, :, :])
                plot_heatmap_pred_sample(x[vis_idx][-1:], y[vis_idx][-1:], y_pred[vis_idx][-1:], c[vis_idx][-1:], bg_mode=param_dict['bg_mode'], save_dir=param_dict['save_dir'], curr_step=step+curr_step, exp_name=exp_name, output_type='jpg')
            else:
                y = to_img_format(y)
                y_pred = to_img_format(y_pred)
                plot_heatmap_pred_sample(x[vis_idx], y[vis_idx], y_pred[vis_idx], c[vis_idx], bg_mode=param_dict['bg_mode'], save_dir=param_dict['save_dir'], curr_step=step+curr_step, exp_name=exp_name)
    
    return float(np.mean(epoch_loss)), step+curr_step

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    if args.gpu == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"TensorBoard: start with 'tensorboard --logdir {os.path.join(args.save_dir, 'logs')}'")
        tb_writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        tb_writer = None

    display_step = 4 if args.debug else args.vis_step
    num_workers = args.batch_size if args.batch_size <= 16 else 16
    
    # Load checkpoint
    if args.resume_training:
        print(f'Load checkpoint from {args.model_name}_cur.pt...')
        assert os.path.exists(os.path.join(args.save_dir, f'{args.model_name}_cur.pt')), f'No checkpoint found in {args.save_dir}'
        ckpt = torch.load(os.path.join(args.save_dir, f'{args.model_name}_cur.pt'))
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
        train_dataset = Shuttlecock_Trajectory_Dataset(split='train', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, frame_alpha=args.frame_alpha, debug=args.debug)
        val_dataset = Shuttlecock_Trajectory_Dataset(split='val', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug)
    elif args.data_type == 'UniBall':
        train_dataset = UniBall_Dataset(split='train', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, frame_alpha=args.frame_alpha, debug=args.debug, heatmap_mode=args.heatmap_mode, SIGMA=args.sigma)
        val_dataset = UniBall_Dataset(split='val', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug, heatmap_mode=args.heatmap_mode, SIGMA=args.sigma)

    if args.dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    print(f'Create {args.model_name}...')
    exp_name = ''
    if args.model_name == 'VideoMamba':
        mamba_args = dict(
            patch_size=args.patch_size,
            d_shallow=args.d_shallow,
            d_model=args.d_model,
            depth=args.depth,
            seq_len=args.seq_len+1,
        )
        model = get_model_videomamba(mamba_args)
        exp_name = f'{args.model_name}_p{args.patch_size}_dp{args.d_model}_ds{args.d_shallow}_depth{args.depth}'
    else:
        model = get_model(args.model_name, args.seq_len, args.bg_mode)
        exp_name = f'{args.model_name}_{args.data_type}_l{args.seq_len}_{data_mode}_{args.bg_mode}_{args.heatmap_mode}'
    
    model = model.to(args.gpu)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params:', n_parameters)
    if args.dist:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)

    # Create optimizer
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optim == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('Invalid optimizer.')

    # Create lr scheduler
    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # Init statistics
    if args.resume_training and args.gpu == 0:
        model.load_state_dict(remove_ddp_prefix(ckpt['model']))
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.lr_scheduler:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        if 'max_val_acc' in ckpt:
            max_val_acc = ckpt['max_val_acc']
        else:
            max_val_acc = 0.
        print(f'Resume training from epoch {start_epoch}...')
        if args.dist:
            dist.broadcast_object_list([model.state_dict(), optimizer.state_dict(), scheduler.state_dict() if scheduler is not None else None], src=0)
    else:
        max_val_acc = 0.
        start_epoch = 0
        

    print(f'Start training...')
    train_start_time = time.time()
    curr_step = 0
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch [{epoch+1} / {args.epochs}]')
        start_time = time.time()
        train_loss, curr_step = train_tracknet(model, optimizer, train_loader, param_dict, exp_name, tb_writer=tb_writer, last_only=args.last_only, curr_step=curr_step, display_step=display_step, gpu=args.gpu)
        
        if args.gpu == 0:
            val_loss, val_res = eval_tracknet(model, val_loader, param_dict)
            write_to_tb(exp_name, tb_writer, (train_loss, val_loss), val_res, epoch, curr_step)
            # Pick best model
            cur_val_acc = val_res['accuracy']
            if cur_val_acc >= max_val_acc:
                max_val_acc = cur_val_acc
                torch.save(dict(epoch=epoch,
                                max_val_acc=max_val_acc,
                                model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict() if scheduler is not None else None,
                                param_dict=param_dict),
                            os.path.join(args.save_dir, f'{args.model_name}_best.pt'))
            
            # Save current model
            torch.save(dict(epoch=epoch,
                            max_val_acc=max_val_acc,
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict() if scheduler is not None else None,
                            param_dict=param_dict),
                        os.path.join(args.save_dir, f'{args.model_name}_cur.pt'))
        
        print(f'Epoch runtime: {(time.time() - start_time) / 3600.:.2f} hrs')
        if args.lr_scheduler:
            scheduler.step()
    
    if tb_writer is not None:
        tb_writer.close()
    print(f'Training time: {(time.time() - train_start_time) / 3600.:.2f} hrs')
    print('Done......')
    if args.dist:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()