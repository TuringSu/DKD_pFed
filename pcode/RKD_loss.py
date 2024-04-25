# encoding：utf-8
# @author: Turing Su
# @time: 2023/6/3 16:43

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 对ABF模块进行定义
class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        # 定义ABF模块的网络层
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channel)
        self.att_conv = None
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 1, kernel_size=1),  # 修改这里的通道数为 1
                nn.Sigmoid()
            )

    def forward(self, x, y=None, shape=None):
        if x.dim() == 2:
            # 处理输入为2维的情况（未进行批处理）
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # 处理输入为3维的情况（已进行批处理，但缺少通道维度）
            x = x.unsqueeze(0)

        n, _, h, w = x.size()

        x = x.to(self.conv1.weight.device)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        if self.att_conv is not None:
            y = F.interpolate(y, (shape, shape), mode="nearest")
            z = torch.cat((x, y), dim=1)
            z = self.att_conv(z)

            x = (x * z + y * (1 - z))  # 修改这里的计算方式
        y = self.conv2(x)

        return y, x


class ReviewKDLoss(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel, shapes=[1, 7, 14, 28, 56], hcl_mode="avg",
                 name="loss_review_kd"):
        super(ReviewKDLoss, self).__init__()
        self.shapes = shapes
        self.name = name
        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1))

        self.abfs = abfs

    def forward(self, student_features, teacher_features):
        x = list(reversed(student_features))
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)

        for idx in range(1, len(x)):

            print(len(self.abfs))
            print(len(self.shapes))
            print(len(x))

            features, abf, shape = x[idx], self.abfs[idx], self.shapes[idx]
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        loss_dict = {}
        loss_dict[self.name] = self.hcl(results, teacher_features)

        return loss_dict


class HCL(nn.Module):
    def __init__(self, mode="avg"):
        super(HCL, self).__init__()
        assert mode in ["max", "avg"]
        self.mode = mode

    def forward(self, fstudent, fteacher):
        loss_all = 0.0

        for fs, ft in zip(fstudent, fteacher):
            h = fs.size(2)
            loss = F.mse_loss(fs, ft)
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                if self.mode == "max":
                    tmpfs = F.adaptive_max_pool2d(fs, (l, l))
                    tmpft = F.adaptive_max_pool2d(ft, (l, l))
                else:
                    tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l, l))

                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft) * cnt
                tot += cnt
            loss = loss / tot
            loss_all += loss
        return loss_all