import os
import torch

import torch.nn.functional as F
import torch.nn as nn

from models.deca.models.encoders import ResnetEncoder
from models.deca.utils import util
from models.deca.utils.config import cfg

torch.backends.cudnn.benchmark = True


class DECA(nn.Module):
    def __init__(self, path=''):
        super(DECA, self).__init__()
        self.cfg = cfg
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.path = path
        self._create_model(self.cfg.model)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam,
                         model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param)
        self.E_detail = ResnetEncoder(outsize=self.n_detail)
        # resume model
        model_path = self.path
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
        else:
            print(f'please check model path: {model_path}')
            # exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    # @torch.no_grad()
    def encode(self, images):
        parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        return codedict

    def forward(self, img):
        img = F.interpolate(img, (224, 224), mode='bilinear')
        codedict = self.encode(img)
        return codedict

    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict()
        }
