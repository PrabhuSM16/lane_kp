import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy


class CenterNet(nn.Module):
  def __init__(self, num_classes, backbone='mobilenet_v3_large', conf_thresh=0.5, bb_last_dim=960, pretrained_backbone=True):
    super(CenterNet, self).__init__()
    self.conf_thresh = conf_thresh
    self.backbone = getattr(tv.models, backbone)(pretrained=pretrained_backbone).features
    up1 = self.upsample(bb_last_dim, 512, 4) # in:960,15,20 out:512,60,80
    up2 = self.upsample(512, 256, 4) # in:512,60,80 out:256,240,320
    self.adapter = nn.Sequential(*[up1,
                                   up2,])

    self.heatmap_head = self.head_module(num_classes)
    self.dimension_head = self.head_module(2)
    self.offset_head = self.head_module(2)
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)

  def forward(self, x):
    x_bb = self.backbone(x)
    x_bb = self.adapter(x_bb)
    x_hmap = self.heatmap_head(x_bb) #raw heatmap
    x_dims = self.dimension_head(x_bb)
    x_offs = self.offset_head(x_bb)

    x_hmap_lm = torch.mul(x_hmap, torch.where(x_hmap==self.maxpool(x_hmap), 1, 0)) #local max keypoints of heatmap
    x_hmap_kp = torch.mul(x_hmap_lm, torch.where(x_hmap_lm>=self.conf_thresh, 1, 0)) #final thresholded keypoints

    return {'x_bb': x_bb,
            'x_hmap': x_hmap,
            'x_hmap_kp': x_hmap_kp,
            'x_dims': x_dims,
            'x_offs': x_offs}

  def upsample(self, cin, cout, sf):
    return nn.Sequential(*[nn.UpsamplingNearest2d(scale_factor=sf),
                            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(cout, eps=0.001, momentum=0.01),
                            nn.Hardswish(),])

  def head_module(self, cout):
    return nn.Sequential(*[nn.Conv2d(256, 128, 3, 2, 1, bias=False),
                           nn.BatchNorm2d(128,eps=0.001,momentum=0.01),
                           nn.Hardswish(),
                           nn.Conv2d(128, cout, 1, 1, 0, bias=False),
                           nn.BatchNorm2d(cout, eps=0.001, momentum=0.01),
                           nn.Hardswish(),])


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = CenterNet(5).to(device)
  net.eval()
  #print(net)

  x = torch.randn(1,3,480,640).to(device)

  with torch.no_grad():
    out = net(x)

  print('Output shapes:')
  print(f'x_bb     : {out["x_bb"].shape}')
  print(f'x_hmap   : {out["x_hmap"].shape}')
  print(f'x_hmap_kp: {out["x_hmap_kp"].shape}')
  print(f'x_dims   : {out["x_dims"].shape}')
  print(f'x_offs   : {out["x_offs"].shape}')
