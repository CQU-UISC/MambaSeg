from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint


from model.encoder.vmamba import Mlp, gMlp, Temporal_SSM, SS2D

  
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]
        x = self.conv1(x)
        return self.sigmoid(x)


class TemporalAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# ====================== Mamba Fusion Block ==========================
class DDIM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        hidden_space_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        # =============================
        mlp_ratio=4.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_space_dim = hidden_space_dim
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm


        self.norm_i = norm_layer(hidden_dim)
        self.norm_e = norm_layer(hidden_dim)

        # self.mixatt = MixAttention(hidden_dim, reduction=4)

        # ========================= CTIM ===========================
        self.cca = CrossTemporalAttention(hidden_dim)

        self.ssm_i = Temporal_SSM(
            # basic dims ===========
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # dwconv ===============
            d_conv=ssm_conv, # < 2 means no conv 
            conv_bias=ssm_conv_bias,
            # ======================
            dropout=ssm_drop_rate,
            # dt init ==============
            initialize=ssm_init,
            # ======================
            forward_type="vc_noz",
            channel_first=channel_first,
            # ======================
            **kwargs,
        )
        self.ssm_e = Temporal_SSM(
            # basic dims ===========
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # dwconv ===============
            d_conv=ssm_conv, # < 2 means no conv 
            conv_bias=ssm_conv_bias,
            # ======================
            dropout=ssm_drop_rate,
            # dt init ==============
            initialize=ssm_init,
            # ======================
            forward_type="vc_noz",
            channel_first=channel_first,
            # ======================
            **kwargs,
        )

        self.conv_i1 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)
        self.conv_e1 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)

        self.ca1 = TemporalAttention(hidden_dim)
        self.ca2 = TemporalAttention(hidden_dim)

        # ========================= CTIM ===========================


        # ========================= CSIM ===========================
        self.csa = CrossSpatialAttention()

        self.ss2d_i = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            initialize=ssm_init,
            # ==========================
            forward_type="v05_noz",
            channel_first=channel_first
        )
        self.ss2d_e = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            initialize=ssm_init,
            # ==========================
            forward_type="v05_noz",
            channel_first=channel_first
        )

        self.conv_i2 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)
        self.conv_e2 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        # ========================= CSIM ===========================

        # 1x1 conv 用来调整通道数
        # 输入: concat(img_c, img_s)(B, 2C, H, W), concat(ev_c, ev_s)(B, 2C, H, W)
        # 输出: img(B, C, H, W), ev(B, C, H, W)
        self.conv_i3 = nn.Conv2d(in_channels = hidden_dim * 2, out_channels = hidden_dim, kernel_size = 1)
        self.conv_e3 = nn.Conv2d(in_channels = hidden_dim * 2, out_channels = hidden_dim, kernel_size = 1)

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def _forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        """
        Args:
            img: [batch_size, hidden_dim, h, w]
            ev: [batch_size, hidden_dim, h, w]
        Process:
            1. Swap the channels between img and ev
            2.1.1 Apply SSM to the swapped channels
            2.1.2 Apply Channel Attention to the swapped channels
            2.2.1 Apply SS2D to the swapped channels
            2.2.2 Apply Spatial Attention to the swapped channels
            3. Concat the results from SSM and SS2D
            4. Apply 1x1 conv to the concatenated results
            5. Apply DropPath to the results
        """

        # Normalize the input tensors
        img_norm = self.norm_i(x_img)
        ev_norm = self.norm_e(x_ev)

        # Cross Temporal Attention
        img_cca, ev_cca = self.cca(img_norm, ev_norm)
        # Cross Spatial Attention
        img_csa, ev_csa = self.csa(img_norm, ev_norm)

        # Cross Temporal SSM
        x_img_c = self.ssm_i(img_cca) + self.conv_i1(img_cca)
        x_ev_c = self.ssm_e(ev_cca) + self.conv_e1(ev_cca)
        
        # Apply Temporal Attention
        x_img_c = x_img_c * self.ca1(x_img_c)   # [12,96,50,87] ca1(x_img_c) [12,96,1,1]
        x_ev_c = x_ev_c * self.ca2(x_ev_c)


        # Cross SS2D
        x_img_s = self.ss2d_i(img_csa) + self.conv_i2(img_csa)
        x_ev_s = self.ss2d_e(ev_csa) + self.conv_e2(ev_csa)

        # Apply Spatial Attention
        x_img_s = x_img_s * self.sa1(x_img_s)   # [12,96,50,87] sa1(x_img_s) [12,1,51,88]
        x_ev_s = x_ev_s * self.sa2(x_ev_s)

        # concat
        x_img_ = torch.cat((x_img_c, x_img_s), dim=1)
        x_ev_ = torch.cat((x_ev_c, x_ev_s), dim=1)
        # Apply 1x1 conv
        x_img_swap = self.conv_i3(x_img_)
        x_ev_swap = self.conv_e3(x_ev_)
        # Apply DropPath
        x_img = self.drop_path1(x_img_swap) + x_img
        x_ev = self.drop_path2(x_ev_swap) + x_ev
        
        return x_img, x_ev


    def forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_img, x_ev)
        else:
            return self._forward(x_img, x_ev)
        

        
class CrossTemporalAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x_img, x_ev):

        x_img, x_ev = self.SwapChannel(x_img, x_ev)

        attn_img = self.shared_fc(x_ev)
        attn_ev = self.shared_fc(x_img)

        x_img_new = x_img * attn_img
        x_ev_new = x_ev * attn_ev
        return x_img_new, x_ev_new
    
    @staticmethod
    def SwapChannel(x_img: torch.Tensor, x_ev: torch.Tensor):
        """
        Swap the channels between img and ev
        Args:
            x_img: [b, c, h, w]
            x_ev: [b, c, h, w]
        Returns:
            swapped_x_img: [b, c, h, w]
            swapped_x_ev: [b, c, h, w]
        """
        # Ensure that both tensors have the same shape
        assert x_img.shape == x_ev.shape, "Both input tensors must have the same shape"
        
        # Get channel indices for odd and even channels
        C = x_img.shape[1]
        odd_indices = torch.arange(1, C, 2).to(x_img.device)
        # even_indices = torch.arange(0, C, 2).to_img(x_img.device)
        
        # Swap odd channels
        swapped_x_img = x_img.clone()
        swapped_x_ev = x_ev.clone()
        
        # For image modality: keep even channels, take event's odd channels
        swapped_x_img[:, odd_indices, :, :] = x_ev[:, odd_indices, :, :]
        
        # For event modality: keep even channels, take image's odd channels
        swapped_x_ev[:, odd_indices, :, :] = x_img[:, odd_indices, :, :]
        
        return swapped_x_img, swapped_x_ev
      
class CrossSpatialAttention_(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(6, 6*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(6*reduction, 3, kernel_size), 
                    nn.Sigmoid())
        
    def forward(self, x_img, x_ev):
        B, _, H, W = x_img.shape

        avg_img = torch.mean(x_img, dim=1, keepdim=True)
        max_img, _ = torch.max(x_img, dim=1, keepdim=True)
        avg_ev = torch.mean(x_ev, dim=1, keepdim=True)
        max_ev, _ = torch.max(x_ev, dim=1, keepdim=True)


        F = x_img + x_ev
        avg_F = torch.mean(F, dim=1, keepdim=True)
        max_F, _ = torch.max(F, dim=1, keepdim=True)


        x_cat = torch.cat((avg_img, max_img, avg_ev, max_ev, avg_F, max_F), dim=1) # B 6 H W
        
        att = self.mlp(x_cat).view(B, 3, 1, H, W)
        # att_img = att[:, 0, :, :, :]
        # att_ev = att[:, 1, :, :, :]
        Att_E = att[:, 0]
        Att_I = att[:, 1]
        Att_F = att[:, 2]
        # x_img_new = x_img * att_ev
        # x_ev_new = x_ev * att_img


        x_img_new = x_img * Att_E * Att_F
        x_ev_new = x_ev * Att_I * Att_F
        return x_img_new, x_ev_new

class CrossSpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(4, 4*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4*reduction, 2, kernel_size), 
                    nn.Sigmoid())
        
    def forward(self, x_img, x_ev):
        B, _, H, W = x_img.shape

        avg_img = torch.mean(x_img, dim=1, keepdim=True)
        max_img, _ = torch.max(x_img, dim=1, keepdim=True)
        avg_ev = torch.mean(x_ev, dim=1, keepdim=True)
        max_ev, _ = torch.max(x_ev, dim=1, keepdim=True)
        
        x_cat = torch.cat((avg_img, max_img, avg_ev, max_ev), dim=1) # B 4 H W
        
        att = self.mlp(x_cat).reshape(B, 2, 1, H, W)
        att_img = att[:, 0, :, :, :]
        att_ev = att[:, 1, :, :, :]
        x_img_new = x_img * att_ev
        x_ev_new = x_ev * att_img
        return x_img_new, x_ev_new