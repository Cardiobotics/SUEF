import torch
import torch.distributed as dist
import numpy as np
import os
from utils.utils import update_cfg

class AverageMeterDDP(object):
    def __init__(self, n_classes=1):
        self.n_classes = n_classes
        self.val = np.zeros(self.n_classes)
        self.avg = np.zeros(self.n_classes)
        self.sum = np.zeros(self.n_classes)
        self.count = 0
        if is_dist_avail_and_initialized():
            self.step = dist.get_world_size()
        else:
            self.step = 1

    def reset(self):
        self.val = np.zeros(self.n_classes)
        self.avg = np.zeros(self.n_classes)
        self.sum = np.zeros(self.n_classes)
        self.count = 0

    def update(self, val):
        if is_dist_avail_and_initialized():
            t_val = torch.tensor(val, dtype=torch.float64, device='cuda')
            dist.barrier()
            dist.all_reduce(t_val)
            val = t_val.cpu().numpy()
        self.count += self.step
        for i in range(0, self.n_classes):
            self.val[i] = val[i]
            self.sum[i] += val[i]
            self.avg[i] = self.sum[i] / self.count


def prepare_ddp(config):
    '''
    Global settings for DDP. NCCL is optimal for cuda usage.
    '''
    update_cfg(config, key='world_size', val=torch.cuda.device_count())
    update_cfg(config, key='dist_backend', val='nccl')
    update_cfg(config, key='dist_url', val='env://')


def init_distributed_mode(config):
    '''
    Starts a DDP process group on localhost
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(config.rank)
    torch.distributed.init_process_group(
        backend=config.dist_backend, init_method=config.dist_url,
        world_size=config.world_size, rank=config.rank)
    print("Started DDP process group at rank {} with world size {} on localhost \n".format(get_rank(), dist.get_world_size()))
    #setup_for_distributed(is_master())


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not the master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    if not is_master:

        def line(*args, **kwargs):
            pass

        def images(*args, **kwargs):
            pass


def cleanup():
    dist.destroy_process_group()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0