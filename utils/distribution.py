import os
import torch
import torch.distributed as dist
import sys

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.local_rank
        args.dist = True
        torch.cuda.set_device(args.gpu)
        torch.distributed.barrier()
        # args.verbose = (args.verbose and args.rank == 0)
        setup_for_distributed(args.gpu == 0)
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.dist = False
    return args

# def init_distributed_mode(args):
#     if args.dist_on_itp:
#         args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
#         args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
#         args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
#         args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['RANK'] = str(args.rank)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = int(os.environ['SLURM_LOCALID'])
#         args.world_size = int(os.environ['SLURM_NTASKS'])
#         os.environ['RANK'] = str(args.rank)
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['WORLD_SIZE'] = str(args.world_size)

#         node_list = os.environ['SLURM_NODELIST']
#         addr = subprocess.getoutput(
#             f'scontrol show hostname {node_list} | head -n1')
#         if 'MASTER_ADDR' not in os.environ:
#             os.environ['MASTER_ADDR'] = addr
#     elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}, gpu {}'.format(
#         args.rank, args.dist_url, args.gpu), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     # assert torch.distributed.is_initialized()
#     setup_for_distributed(args.rank == 0)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
