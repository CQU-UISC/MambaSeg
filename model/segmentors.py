from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from model.encoder.vmamba import Backbone_VSSM, Permute, LayerNorm2d
from model.encoder.fuse_modules import DDIM


class MambaSeg(nn.Module):
    def __init__(self, ver_img='vmamba_tiny', ver_ev='vmamba_tiny', num_classes=6, 
                 fuse=None, num_channels_img=3, pretrained_img=None, 
                 num_channels_ev=3, pretrained_ev=None, data_type=None, img_size=None, if_viz=False):
        super().__init__()
        self.num_channels_img = num_channels_img
        self.num_channels_ev = num_channels_ev
        self.out_indices=(0, 1, 2, 3)

        dim_dict = { 'vmamba_tiny': [96, 192, 384, 768],
                        'vmamba_small': [96, 192, 384, 768],
                        'vmamba_base': [128, 256, 512, 1024]}

        patchembed_version = "v2"
        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )        

        self.patch_size = 4
        self.patch_norm = True
        self.norm_layer = "ln2d"
        self.channel_first = (self.norm_layer.lower() in ["bn", "ln2d"])

        dim_img = dim_dict[ver_img]
        dim_ev = dim_dict[ver_ev]
        # ceiling
        space_dim = [(ceil(img_size[0] / self.patch_size), ceil(img_size[1] / self.patch_size))]
        for i in range(len(dim_img)-1):
            space_dim.append((ceil(space_dim[i][0] / 2), ceil(space_dim[i][1] / 2)))
        self.space_dim = space_dim  # (H, W) for each stage
        self.num_tokens = [space_dim[i][0] * space_dim[i][1] for i in range(len(self.space_dim))]  # number of tokens for each stage

        # Select an encoder 设置图像-事件双分支的编码器
        
        self.encoder_img = Backbone_VSSM(patch_size=4, in_chans=3, num_classes=1000, 
                                         depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
                                         ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", 
                                         ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", 
                                         mlp_ratio=4.0,downsample_version="v3", patchembed_version="v2", 
                                         norm_layer=self.norm_layer, use_checkpoint=False,
                                         out_indices=self.out_indices, pretrained=pretrained_img)
        

        self.encoder_ev = Backbone_VSSM(patch_size=4, in_chans=3, num_classes=1000, 
                                         depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
                                         ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", 
                                         ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", 
                                         mlp_ratio=4.0,downsample_version="v3", patchembed_version="v2", 
                                         norm_layer=self.norm_layer, use_checkpoint=False,
                                         out_indices=self.out_indices, pretrained=pretrained_ev)

        # Replace the first layer of the encoder with a channel-adaptive layer
        self.norm_layer: nn.Module = _NORMLAYERS.get(self.norm_layer.lower(), None)
        if self.num_channels_img != 3:
            self.encoder_img.patch_embed = _make_patch_embed(self.num_channels_img, dim_img[0], self.patch_size, self.patch_norm, self.norm_layer, channel_first=self.channel_first)


        self.encoder_ev.patch_embed = _make_patch_embed(self.num_channels_ev, dim_ev[0], self.patch_size, self.patch_norm, self.norm_layer, channel_first=self.channel_first)


        self.augs = nn.ModuleList()
        self.ccbs = nn.ModuleList()

        # 添加每个Stage的融合模块
        for i in range(len(dim_img)):
            self.ccbs.append(
                DDIM(
                    hidden_dim=dim_img[i],
                    hidden_space_dim=self.num_tokens[i],
                    norm_layer=LayerNorm2d,
                    channel_first=True,
                )
            )

        self.out_convs = nn.ModuleList([
            nn.Conv2d(in_channels = dim_img[i] * 2, 
                      out_channels = dim_img[i], 
                      kernel_size = 1)
            for i in range(len(dim_img))
        ])
        

        # Decoder
        from model.decoder.segformer_decoder import SegFormerHead
        self.decoder = SegFormerHead(in_channels=dim_img, num_classes=num_classes)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    
    def forward(self, x_ev, x_img):

        return self.forward_fuse_v2(x_ev, x_img)
    
    def forward_fuse_v2(self, x_ev, x_img):
        B, _, H0, W0 = x_ev.shape
        outs = []

        """ Encode the event-image fusion representation """
        # patch_embed
        x_ev = self.encoder_ev.patch_embed(x_ev)
        x_img = self.encoder_img.patch_embed(x_img)
        outs_ev, outs_img = [], []
        # loop encoder_ev and encoder_img
        for i, (layer_ev, layer_img) in enumerate(zip(self.encoder_ev.layers, self.encoder_img.layers)):
            # Get Stage i output
            o_ev = layer_ev.blocks(x_ev)
            o_img = layer_img.blocks(x_img)
            if i in self.out_indices:
                # norm
                norm_layer_ev = getattr(self.encoder_ev, f'outnorm{i}')
                norm_layer_img = getattr(self.encoder_img, f'outnorm{i}')
                out_ev = norm_layer_ev(o_ev)
                out_img = norm_layer_img(o_img)
                # add to backbone outs
                outs_ev.append(out_ev.contiguous())
                outs_img.append(out_img.contiguous())


            # fuse
            img, ev = self.ccbs[i](outs_img[i], outs_ev[i])
            # add fuse out to backbone next stage input
            x_ev = o_ev + ev
            x_img = o_img + img
            x_ev = layer_ev.downsample(x_ev)
            x_img = layer_img.downsample(x_img)
            concat_out = torch.cat((img, ev), dim=1)
            outs.append(self.out_convs[i](concat_out))

        # outs
        # print("len outs:", len(outs))
        # for out in outs:
            # print("out shape:", out.shape)

        """ Generate the segmentation mask """
        x = self.decoder(outs)
        x = F.interpolate(x, size=[H0, W0], mode='bilinear', align_corners=False)
        return x