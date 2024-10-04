import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

from .MiT import mit_b2
from .MobileNetv2 import mobilenet_v2
from torch import einsum
from einops import rearrange
from .Transpose_head import ProjectionHead

class pp_upsample(nn.Module):
    def __init__(self, inc, outc, con_select=True, up=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.PReLU()
        )
        self.con_select = con_select
        self.up = up
    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        if self.con_select and self.up:
            return self.conv(p)
        elif self.con_select and not self.up:
            return self.conv(x)
        else:
            return p

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class SalHead(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, n_classes, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out


class Fusion(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=7):
        super(Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tp = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.ca = ChannelAttention(out_dim)

    def forward(self, x1, x2):
        # max_out, _ = torch.max(x2, dim=1, keepdim=True)
        mean_out = torch.mean(x2, dim=1, keepdim=True)
        x2 = mean_out
        att2 = self.sigmoid(x2)
        out = torch.mul(x1, att2) + x1 + x2
        tp = self.tp(out)
        fuseout = self.ca(tp)

        return fuseout

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)

        return self.relu(bottle)


class FilterModule(nn.Module):
    def __init__(self, Ch, h, window):
        super().__init__()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            print(cur_window, cur_head_split, padding_size)
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.LP = LowPassModule(Ch * h)

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H = size[0]
        W = size[1]

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        LP = self.LP(v_img)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        HP_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        HP = torch.cat(HP_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        HP = rearrange(HP, "B (h Ch) H W -> B h (H W) Ch", h=h)
        LP = rearrange(LP, "B (h Ch) H W -> B h (H W) Ch", h=h)

        dynamic_filters = q * HP + LP
        return dynamic_filters


class Frequency_FilterModule(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = FilterModule(Ch=dim//num_heads, h=num_heads, window={3: 2, 5: 3, 7: 3})

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)
        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SpatialAttention_max(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_max, self).__init__()
        self.inc = torch.tensor(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_spatial = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction1, in_channels, bias=False),
        )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )



    def forward(self, x):

        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)

        y_spatial = self.fc_spatial(y_avg).view(b, c, 1, 1)
        y_channel = self.fc_channel(y_avg).view(b, c, 1, 1)
        y_channel = y_channel.sigmoid()

        map = (x * (y_spatial)).sum(dim=1) / self.inc
        map = (map / self.inc).sigmoid().unsqueeze(dim=1)
        return map * x * y_channel




class EnDecoderModel(nn.Module):
    def __init__(self, n_classes, backbone):
        super(EnDecoderModel, self).__init__()
        if backbone == 'segb2R_mobilev2D':
            self.backboner = mit_b2()
            self.backboned = mobilenet_v2(pretrained=True)
        Highdims = [64, 128, 320, 512]  #backboner_dims
        Lowdims = [24, 32, 160, 320]    #backboned_dims
        ###################Feature_Align##########################
        self.align_convF4 = pp_upsample(Lowdims[3], Highdims[3], con_select=True, up=False)
        self.align_convF3 = pp_upsample(Lowdims[2], Highdims[2], con_select=True, up=False)
        self.align_convF2 = pp_upsample(Lowdims[1], Highdims[1], con_select=True, up=False)
        self.align_convF1 = pp_upsample(Lowdims[0], Highdims[0], con_select=True, up=False)
        #############################################
        self.up_convF4_F3 = pp_upsample(Highdims[3], Highdims[2])
        self.up_convF3_F2 = pp_upsample(Highdims[2], Highdims[1])
        self.up_convF2_F1 = pp_upsample(Highdims[1], Highdims[0])
        self.up_convF1_nclass = pp_upsample(Highdims[0], n_classes)
        self.onlyout_up2 = pp_upsample(Highdims[0], Highdims[0], con_select=False)

        self.F34_p = SalHead(Highdims[2], n_classes)
        self.F234_p = SalHead(Highdims[1], n_classes)
        self.F1234_p = SalHead(Highdims[0], n_classes)
        self.out_drop = SalHead(n_classes, n_classes)

        self.fuse1 = Fusion(Highdims[0], Highdims[0])
        self.fuse2 = Fusion(Highdims[1], Highdims[1])
        self.fuse3 = Fusion(Highdims[2], Highdims[2])
        self.fuse4 = Fusion(Highdims[3], Highdims[3])
        self.ff1 = Frequency_FilterModule(Highdims[0])
        self.ff2 = Frequency_FilterModule(Highdims[1])
        self.ff3 = Frequency_FilterModule(Highdims[2])
        self.ff4 = Frequency_FilterModule(Highdims[3])

        self.F4_contra = SalHead(Highdims[3], n_classes)
        self.contrative_head = ProjectionHead(dim=n_classes)

    def forward(self, rgb, dep):

        # with torch.no_grad():
        #     prompt_r = self.backbone(rgb)
        #     features_prompt_rlist = prompt_r[0]
        #     pro_rf1 = features_prompt_rlist[0]
        #     pro_rf2 = features_prompt_rlist[1]
        #     pro_rf3 = features_prompt_rlist[2]
        #     pro_rf4 = features_prompt_rlist[3]
        df1 = self.backboned.features[0:4](dep)
        df2 = self.backboned.features[4:7](df1)
        df3 = self.backboned.features[7:17](df2)
        df4 = self.backboned.features[17:18](df3)

        rf1 = self.backboner.layer1(rgb)
        #############################################
        # Encoder #
        #############################################
        FD1_ad = rf1 + self.align_convF1(df1)
        FD1_fuse = self.fuse1(rf1, self.align_convF1(df1))
        # B_1, C_1, H_1, W_1 = FD1.size()
        # FD1_fuse_aff = self.ff1(FD1_fuse.permute(0, 2, 3, 1).view(B_1, -1, C_1), [H_1, W_1]).permute(0, 2, 1).view(B_1, C_1, H_1, W_1)
        # FD1_aff = self.ff1(FD1.permute(0, 2, 3, 1).view(B_1, -1, C_1), [H_1, W_1]).permute(0, 2, 1).view(B_1, C_1, H_1, W_1)
        #############################################
        rf2 = self.backboner.layer2(FD1_ad)
        #############################################
        FD2_ad = rf2 + self.align_convF2(df2)
        FD2_fuse = self.fuse2(rf2, self.align_convF2(df2))
        # B_2, C_2, H_2, W_2 = FD2.size()
        # FD2_fuse_aff = self.ff2(FD2_fuse.permute(0, 2, 3, 1).view(B_2, -1, C_2), [H_2, W_2]).permute(0, 2, 1).view(B_2, C_2, H_2, W_2)
        # FD2_aff = self.ff2(FD2.permute(0, 2, 3, 1).view(B_2, -1, C_2), [H_2, W_2]).permute(0, 2, 1).view(B_2, C_2, H_2, W_2)
        #############################################
        rf3 = self.backboner.layer3(FD2_ad)
        #############################################
        FD3_ad = rf3 + self.align_convF3(df3)
        FD3_fuse = self.fuse3(rf3, self.align_convF3(df3))

        # FD3_fuse = self.fuse3(rf3, self.align_convF3(df3))
        B_3, C_3, H_3, W_3 = FD3_ad.size()
        FD3_ad_aff = self.ff3(FD3_ad.permute(0, 2, 3, 1).view(B_3, -1, C_3), [H_3, W_3]).permute(0, 2, 1).view(B_3, C_3, H_3, W_3)
        #############################################
        rf4 = self.backboner.layer4(FD3_ad)
        #############################################
        FD4_ad = rf4 + self.align_convF4(df4)
        FD4_fuse = self.fuse4(rf4, self.align_convF4(df4))
        # FD4_fuse = self.fuse4(rf4, self.align_convF4(df4))
        B_4, C_4, H_4, W_4 = FD4_ad.size()
        FD4_ad_aff = self.ff4(FD4_ad.permute(0, 2, 3, 1).view(B_4, -1, C_4), [H_4, W_4]).permute(0, 2, 1).view(B_4, C_4, H_4, W_4)
        ##############################################
        emb_P4 = self.contrative_head(self.F4_contra(FD4_ad))
        FD_pervise = []
        FD4_up2_DScon3 = self.up_convF4_F3(FD4_fuse + FD4_ad_aff)
        # FD4_up2_DScon3 = self.up_convF4_F3(FD4_aff)
        # FD34 = FD3_aff + FD4_up2_DScon3
        # B_3, C_3, H_3, W_3 = FD3_fuse.size()
        # FD3_fuse_aff = self.ff3(FD3_fuse.permute(0, 2, 3, 1).view(B_3, -1, C_3), [H_3, W_3]).permute(0, 2, 1).view(B_3, C_3, H_3, W_3)
        FD34 = FD3_ad_aff + FD3_fuse + FD4_up2_DScon3
        # FD34 = self.fuse3(FD3, FD4_up2_DScon3)


        FD34_p = self.F34_p(FD34)
        FD_pervise.append(FD34_p)


        FD34_up2_DScon3 = self.up_convF3_F2(FD34)
        # FD234 = FD34_up2_DScon3 + FD2
        B_2, C_2, H_2, W_2 = FD2_fuse.size()
        FD2_fuse_aff = self.ff2(FD2_fuse.permute(0, 2, 3, 1).view(B_2, -1, C_2), [H_2, W_2]).permute(0, 2, 1).view(B_2,
                                                                                                                   C_2,
                                                                                                                   H_2,
                                                                                                                   W_2)
        FD234 = FD34_up2_DScon3 + FD2_fuse_aff



        FD234_p = self.F234_p(FD234)
        FD_pervise.append(FD234_p)

        FD234_up2_DScon3 = self.up_convF2_F1(FD234)
        # FD1234 = FD234_up2_DScon3 + FD1
        B_1, C_1, H_1, W_1 = FD1_fuse.size()
        FD1_fuse_aff = self.ff1(FD1_fuse.permute(0, 2, 3, 1).view(B_1, -1, C_1), [H_1, W_1]).permute(0, 2, 1).view(B_1,
                                                                                                                   C_1,
                                                                                                                   H_1,
                                                                                                                   W_1)
        FD1234 = FD234_up2_DScon3 + FD1_fuse_aff

        FD1234_p = self.F1234_p(FD1234)
        FD_pervise.append(FD1234_p)

        out = self.up_convF1_nclass(FD1234)
        out = self.onlyout_up2(out)
        out_upinit_DSnumclass = self.out_drop(out)

        return out_upinit_DSnumclass, FD_pervise, emb_P4

    def load_pre_b2(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backboner.load_state_dict(new_state_dict3, strict=False)
        print('B2.Pth_backboner loading')


