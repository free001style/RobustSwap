import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2.model import EqualLinear
from models.psp.psp_encoders import Inverter
from utils.torch_utils import get_keys, requires_grad
from models.deca.deca import DECA
from models.stylegan2.model import Generator


class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.backbone = nn.ModuleList()
        self.backbone.append(EqualLinear(100, 512))
        for i in range(4):
            self.backbone.append(EqualLinear(512, 512))
        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, alpha):
        """alpha -- B x 100
            return B x 512"""
        return self.backbone(alpha)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Conv2d(3, 64, 3, padding=1))
        self.backbone.append(nn.LeakyReLU(0.2))
        self.backbone.append(nn.Conv2d(64, 128, 3, padding=1))
        self.backbone.append(nn.LeakyReLU(0.2))
        in_c = 128
        out_c = 256
        for i in range(3):
            self.backbone.append(nn.Conv2d(in_c, out_c, 3, padding=1, stride=2))
            self.backbone.append(nn.LeakyReLU(0.2))
            self.backbone.append(nn.Conv2d(out_c, out_c, 3, padding=1))
            if i < 2:
                self.backbone.append(nn.LeakyReLU(0.2))
            in_c = out_c
            if i == 1:
                out_c *= 2
        self.backbone = nn.Sequential(*self.backbone)
        self.img_conv = nn.Conv2d(512, 3, 1, padding=0)

    def forward(self, img, return_img=True):
        """img -- batch x 3 x 256 x 256
            return -- batch x 512 x 32 x 32"""
        feat = self.backbone(img)
        if return_img:
            return feat, self.img_conv(feat)
        return feat, None


class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()
        self.target_encoder = Encoder()

        self.source_identity = Inverter(opts)
        ckpt = torch.load(opts.sfe_inverter_path)
        self.source_identity.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
        requires_grad(self.source_identity, False)

        self.source_shape = DECA(opts.deca_path)
        requires_grad(self.source_shape, False)

        self.mapping = Mapper()

        self.G = Generator(1024, 512, 8)

        if not opts.train_G:
            requires_grad(self.G, False)
        # notice that the 8-layer fully connected module is always fixed
        else:
            requires_grad(self.G.style, False)

        requires_grad(self.G.input, False)
        requires_grad(self.G.conv1, False)
        requires_grad(self.G.to_rgb1, False)
        requires_grad(self.G.convs[:6], False)
        requires_grad(self.G.to_rgbs[:3], False)

    def forward(self, source, target):
        """source -- B x 3 x 1024 x 1024
           target -- B x 3 x 1024 x 1024
           skip -- B x 3 x 32 x 32"""
        s_256 = F.interpolate(source, (256, 256), mode='bilinear')
        t_256 = F.interpolate(target, (256, 256), mode='bilinear')

        s_w_id, _ = self.source_identity.fs_backbone(s_256)
        alpha = self.source_shape(source)['shape']
        s_w_shape = self.mapping(alpha)
        s_feat = s_w_id + s_w_shape[:, None, :] + self.latent_avg[None, ...]

        t_feat, rgb_image = self.target_encoder(t_256)

        img, _ = self.G([s_feat], t_feat, rgb_image)
        return img
