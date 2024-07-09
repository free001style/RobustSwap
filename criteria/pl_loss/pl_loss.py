import torch
from torch import nn
from models.deca.deca import DECA
from criteria.pl_loss.flame import FLAME
from models.deca.utils.config import cfg


class PLLoss(nn.Module):
    def __init__(self, opts):
        super(PLLoss, self).__init__()
        self.net = DECA(opts.deca_path)
        self.flame = FLAME(cfg, opts.flame_path)

        self.set_requires_grad(False)

    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def forward(self, source, target, swap):
        source_params = self.net(source)
        target_params = self.net(target)
        swap_params = self.net(swap)
        vert_true = self.flame(shape_params=source_params['shape'], expression_params=target_params['exp'],
                               pose_params=target_params['pose'])
        vert_fake = self.flame(shape_params=swap_params['shape'], expression_params=swap_params['exp'],
                               pose_params=swap_params['pose'])
        return torch.abs(vert_true - vert_fake).mean()
