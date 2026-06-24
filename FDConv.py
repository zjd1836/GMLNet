import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from thop import profile, clever_format

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from torch.utils.checkpoint import checkpoint # 导入torch.utils.checkpoint，用于梯度检查点，可以节省显存

# from mmcv.cnn import CONV_LAYERS # 被注释掉的行，可能用于MMCV框架的注册机制，此处未启用
from torch import Tensor # 导入Tensor类型，用于类型提示
import torch.nn.functional as F # 再次导入F，可能习惯性多写一次
import math # 导入math模块，用于数学运算，如log2

# from timm.models.layers import trunc_normal_ # 被注释掉的行，可能用于timm库的权重初始化方法，此处未启用


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    这是一个自定义的激活函数，其形式为： scale * ReLU(x)^2 + bias。
    论文中提到 KSM 的全局分支使用了 StarReLU 作为激活函数。
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        """
        初始化 StarReLU 模块。

        Args:
            scale_value (float): 初始的缩放因子 s。
            bias_value (float): 初始的偏置 b。
            scale_learnable (bool): 缩放因子 s 是否可学习。
            bias_learnable (bool): 偏置 b 是否可学习。
            mode (str, optional): 未使用的参数，可能用于未来的扩展。
            inplace (bool): 是否在原地执行 ReLU 操作（节省内存）。
        """
        super().__init__() # 调用父类 nn.Module 的构造函数
        self.inplace = inplace # 保存 inplace 参数
        self.relu = nn.ReLU(inplace=inplace) # 创建一个 ReLU 激活函数实例
        # 创建可学习的缩放因子和偏置，使用 nn.Parameter 包装，使其成为模型参数
        self.scale = nn.Parameter(scale_value * torch.ones(1), # scale_value * torch.ones(1) 创建一个值为 scale_value 的张量
                                  requires_grad=scale_learnable) # 根据 scale_learnable 决定是否可学习
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        """
        前向传播函数。
        Args:
            x (Tensor): 输入张量。
        Returns:
            Tensor: 经过 StarReLU 激活后的输出张量。
        """
        return self.scale * self.relu(x) ** 2 + self.bias # 实现 s * relu(x)^2 + b 的计算


class KernelSpatialModulation_Global(nn.Module):
    """
    KernelSpatialModulation_Global: KSM 的全局分支。
    此模块负责从输入特征图中提取全局信息，并生成用于调制卷积核的注意力权重。
    它预测了通道注意力、滤波器注意力、空间注意力以及核（并行权重）注意力。
    """
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0,
                 ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        """
        初始化 KernelSpatialModulation_Global 模块。

        Args:
            in_planes (int): 输入特征图的通道数。
            out_planes (int): 输出特征图的通道数（用于滤波器注意力）。
            kernel_size (int): 卷积核的空间大小（例如，3x3 的卷积核，kernel_size=3）。
            groups (int): 卷积的组数（用于判断是否为深度可分离卷积）。
            reduction (float): 注意力分支中用于降维的缩减率。
            kernel_num (int): 并行卷积核的数量（例如，FDConv 中的 n）。
            min_channel (int): 注意力分支降维后的最小通道数。
            temp (float): 注意力计算中的温度参数，用于控制 sigmoid/softmax 的平滑度。
            kernel_temp (float, optional): 核注意力计算的独立温度参数，如果为 None 则使用 temp。
            kernel_att_init (str): 核注意力层的初始化策略，例如 'dyconv_as_extra'。
            att_multi (float): 注意力输出的乘法因子，用于放大或缩小注意力值。
            ksm_only_kernel_att (bool): 是否只输出核注意力，跳过通道、滤波器、空间注意力。
            att_grid (int): 未使用的参数，可能用于未来的网格注意力或空间注意力细化。
            stride (int): 卷积的步长（用于滤波器注意力，影响输出特征图的空间大小）。
            spatial_freq_decompose (bool): 是否进行空间频率分解。如果为 True，通道/滤波器注意力输出通道数可能翻倍。
            act_type (str): 注意力输出的激活函数类型 ('sigmoid', 'tanh', 'softmax')。
        """
        super(KernelSpatialModulation_Global, self).__init__() # 调用父类 nn.Module 的构造函数
        # 计算注意力分支的中间通道数，确保不小于 min_channel
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type # 保存激活函数类型
        self.kernel_size = kernel_size # 保存卷积核大小
        self.kernel_num = kernel_num # 保存并行核数量

        self.temperature = temp # 保存主温度参数
        # 核注意力温度参数，如果没有独立设置则使用主温度参数
        self.kernel_temp = kernel_temp if kernel_temp is not None else temp

        self.ksm_only_kernel_att = ksm_only_kernel_att # 是否只计算核注意力
        self.kernel_att_init = kernel_att_init # 保存核注意力初始化策略
        self.att_multi = att_multi # 保存注意力乘法因子

        self.avgpool = nn.AdaptiveAvgPool2d(1) # 全局平均池化，将空间维度变为 1x1，用于聚合全局信息
        self.att_grid = att_grid # 未使用
        # 降维的1x1卷积，用于从输入通道生成注意力特征
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel) # 批归一化
        self.relu = StarReLU() # 使用自定义的 StarReLU 激活函数

        self.spatial_freq_decompose = spatial_freq_decompose # 保存空间频率分解标志

        # 通道注意力函数定义
        if ksm_only_kernel_att:
            self.func_channel = self.skip # 如果只关注核注意力，则通道注意力直接跳过（返回1.0）
        else:
            # 根据是否进行空间频率分解，决定通道注意力输出的通道数
            # 如果是空间频率分解且核大小大于1，则通道注意力需要为每个频率分量生成调制，所以通道数翻倍
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1, bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention # 否则使用 get_channel_attention 方法

        # 滤波器注意力函数定义
        # 如果是深度可分离卷积（in_planes == groups and in_planes == out_planes）
        # 或者只关注核注意力时，跳过滤波器注意力
        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            # 同理，根据是否进行空间频率分解，决定滤波器注意力输出的通道数
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        # 空间注意力函数定义
        # 如果是 1x1 卷积（点卷积）时，没有空间维度可以调制，或者只关注核注意力时，跳过空间注意力
        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            # 空间注意力输出的通道数等于卷积核的空间元素数量 (kernel_size * kernel_size)
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # 核注意力函数定义
        # 如果只有一个核，则核注意力没有意义，跳过
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            # 核注意力输出的通道数等于并行核的数量
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights() # 调用权重初始化函数

    def _initialize_weights(self):
        """
        初始化模块中的所有卷积层和批归一化层的权重。
        默认使用 Kaiming 正态分布初始化卷积权重，偏置为0。
        批归一化层的权重为1，偏置为0。
        针对特定层，如果 `kernel_att_init` 不是 'dyconv_as_extra'，则可能进行额外的全零或小值初始化（被注释掉的部分）。
        """
        for m in self.modules(): # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d): # 如果是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 使用 Kaiming 初始化，适用于 ReLU 激活
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 偏置初始化为0
            if isinstance(m, nn.BatchNorm2d): # 如果是批归一化层
                nn.init.constant_(m.weight, 1) # 权重初始化为1
                nn.init.constant_(m.bias, 0) # 偏置初始化为0

        # 以下是被注释掉的初始化代码，表明在开发过程中可能尝试过多种精细化初始化策略
        # if hasattr(self, 'channel_spatial'): # 如果存在 channel_spatial 属性
        #     nn.init.normal_(self.channel_spatial.conv.weight, std=1e-6) # 使用小标准差的正态分布初始化
        # if hasattr(self, 'filter_spatial'):
        #     nn.init.normal_(self.filter_spatial.conv.weight, std=1e-6)

        # 对空间注意力层进行初始化
        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            # nn.init.constant_(self.spatial_fc.weight, 0) # 尝试过全零初始化权重
            nn.init.normal_(self.spatial_fc.weight, std=1e-6) # 使用小标准差的正态分布初始化权重
            # self.spatial_fc.weight *= 1e-6 # 进一步缩小权重值
            if self.kernel_att_init == 'dyconv_as_extra': # 如果初始化策略是 'dyconv_as_extra'
                pass # 保持当前初始化（通常是Kaiming或上述的normal_）
            # else:
                # nn.init.constant_(self.spatial_fc.weight, 0)
                # nn.init.constant_(self.spatial_fc.bias, 0)

        # 对滤波器注意力层进行初始化
        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d): # 注意这里是 self.func_filter 而不是 self.filter_fc
            # nn.init.constant_(self.func_filter.weight, 0)
            nn.init.normal_(self.func_filter.weight, std=1e-6)
            # self.func_filter.weight *= 1e-6
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            # else:
                # nn.init.constant_(self.func_filter.weight, 0)
                # nn.init.constant_(self.func_filter.bias, 0)

        # 对核注意力层进行初始化
        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            # nn.init.constant_(self.kernel_fc.weight, 0)
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
                # nn.init.constant_(self.kernel_fc.weight, 0) # 特殊初始化：权重为0，偏置为-10
                # nn.init.constant_(self.kernel_fc.bias, -10)
                # nn.init.constant_(self.kernel_fc.weight[0], 6) # 特殊初始化：第一个核的权重为6，其他为-6
                # nn.init.constant_(self.kernel_fc.weight[1:], -6)
            # else:
                # nn.init.constant_(self.kernel_fc.weight, 0)
                # nn.init.constant_(self.kernel_fc.bias, 0)
                # nn.init.constant_(self.kernel_fc.bias, -10)
                # nn.init.constant_(self.kernel_fc.bias[0], 10)

        # 对通道注意力层进行初始化
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            # nn.init.constant_(self.channel_fc.weight, 0)
            nn.init.normal_(self.channel_fc.weight, std=1e-6)
            # nn.init.constant_(self.channel_fc.bias[1], 6)
            # nn.init.constant_(self.channel_fc.bias, 0)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            # else:
                # nn.init.constant_(self.channel_fc.weight, 0)
                # nn.init.constant_(self.channel_fc.bias, 0)

    def update_temperature(self, temperature):
        """
        更新注意力计算中的温度参数。
        """
        self.temperature = temperature

    @staticmethod
    def skip(_):
        """
        一个静态方法，当不需要计算某种注意力时，返回一个常数1.0作为占位符。
        """
        return 1.0

    def get_channel_attention(self, x):
        """
        计算通道注意力。
        Args:
            x (Tensor): 经过FC层、BN层和StarReLU激活函数处理后的全局特征，形状为 (B, attention_channel, H, W)。
        Returns:
            Tensor: 通道注意力张量。
                    其形状为 (B, 1, 1, Cin, H, W) 或 (B, 1, 1, Cin*2, H, W) （如果spatial_freq_decompose为True），
                    用于与卷积核的输入通道维度进行元素级乘法。
        """
        # self.channel_fc(x) 的输出形状为 (B, in_planes, H, W) 或 (B, in_planes*2, H, W)。
        # .view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) 将通道维度（in_planes或in_planes*2）
        # 映射到第四个维度（索引为3），使其能与卷积核的 (Cin, Cout, K, K) 维度中的 Cin 对齐。
        if self.act_type == 'sigmoid':
            # 应用 sigmoid 激活，除以温度参数，再乘以 att_multi 进行缩放
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            # 应用 tanh 激活，除以温度参数，加1
            channel_attention = 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError # 如果 act_type 不支持，则抛出错误
        return channel_attention

    def get_filter_attention(self, x):
        """
        计算滤波器注意力。
        Args:
            x (Tensor): 经过FC层、BN层和StarReLU激活函数处理后的全局特征，形状为 (B, attention_channel, H, W)。
        Returns:
            Tensor: 滤波器注意力张量。
                    其形状为 (B, 1, Cout, 1, H, W) 或 (B, 1, Cout*2, 1, H, W) （如果spatial_freq_decompose为True），
                    用于与卷积核的输出通道维度进行元素级乘法。
        """
        # self.filter_fc(x) 的输出形状为 (B, out_planes, H, W) 或 (B, out_planes*2, H, W)。
        # .view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) 将通道维度（out_planes或out_planes*2）
        # 映射到第三个维度（索引为2），使其能与卷积核的 (Cin, Cout, K, K) 维度中的 Cout 对齐。
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        """
        计算空间注意力。
        Args:
            x (Tensor): 经过FC层、BN层和StarReLU激活函数处理后的全局特征，形状为 (B, attention_channel, H, W)。
        Returns:
            Tensor: 空间注意力张量，形状为 (B, 1, 1, 1, K, K)，
                    用于与卷积核的空间维度 (K, K) 进行元素级乘法。
        """
        # self.spatial_fc(x) 的输出形状为 (B, kernel_size * kernel_size, H, W)。
        # .view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) 将扁平化的空间维度
        # 重新整形为 (K, K) 并映射到最后两个维度。
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        """
        计算核注意力（即 FDConv 中用于混合 n 个并行核的注意力权重）。
        Args:
            x (Tensor): 经过FC层、BN层和StarReLU激活函数处理后的全局特征，形状为 (B, attention_channel, H, W)。
        Returns:
            Tensor: 核注意力张量，形状为 (B, kernel_num, 1, 1, 1, 1)，
                    用于与 n 个并行卷积核进行加权求和。
        """
        # self.kernel_fc(x) 的输出形状为 (B, kernel_num, H, W)。
        # .view(x.size(0), -1, 1, 1, 1, 1) 将通道维度（kernel_num）映射到第二个维度（索引为1），
        # 并将其他维度压缩为1，以便于对并行核进行加权。
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1) # -1 表示 kernel_num
        if self.act_type == 'softmax':
            # softmax 用于确保所有核的注意力权重和为1，通常用于加权求和的场景
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            # sigmoid 激活后，除以核数，可能是一种归一化尝试，使其总和接近2
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            # tanh 激活后，加1再除以核数进行归一化，使其总和接近2
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
        return kernel_attention

    def forward(self, x, use_checkpoint=False):
        """
        前向传播函数。
        Args:
            x (Tensor): 输入特征图。
            use_checkpoint (bool): 是否使用梯度检查点（可以节省显存但增加计算时间）。
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: 分别是通道注意力、滤波器注意力、空间注意力、核注意力。
        """
        if use_checkpoint:
            # 如果启用检查点，则使用 checkpoint 函数包装 _forward 方法
            # 这会牺牲一部分计算时间来减少反向传播时的显存占用
            return checkpoint(self._forward, x)
        else:
            return self._forward(x) # 否则直接调用 _forward 方法

    def _forward(self, x):
        """
        实际的前向传播逻辑。
        """
        # 输入 x 经过 fc 层、bn 层和 StarReLU 激活函数，得到用于生成注意力的全局特征表示
        # 这里的 avg_x 实际上并不是全局平均池化后的 1x1 特征，而是经过 fc、bn、relu 后的张量，
        # 它的空间维度可能与输入 x 相同，或者在 fc 中被压缩。
        # 从 fc 是 Conv2d(in_planes, attention_channel, 1, bias=False) 来看，它不改变空间维度。
        # 因此，avg_x 仍保留空间维度，并且后续的 attention_fc 层会利用这些空间信息。
        # 尽管名字是 KernelSpatialModulation_Global，但它的注意力计算是基于空间变化的。
        avg_x = self.relu(self.bn(self.fc(x)))
        # 分别调用各个注意力生成函数，并返回它们的结果
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # 以下是被注释掉的备选前向传播逻辑，表明在开发过程中尝试过不同的特征处理和注意力聚合方式
        # return self.attup.flow_warp(self.func_channel(x), grid), self.attup.flow_warp(self.func_filter(x), grid), self.func_spatial(avg_x), self.func_kernel(avg_x), sp_gate
        # return (self.func_channel(x_h) * self.func_channel(x_w)).sqrt(), (self.func_filter(x_h) * self.func_filter(x_w)).sqrt(), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.func_channel(x_h) * self.func_channel(x_w)), (self.func_filter(x_h) * self.func_filter(x_w)), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return ((self.func_channel(x_h) + self.func_channel(x_w)) * csg).sigmoid_() * self.att_multi, ((self.func_filter(x_h) + self.func_filter(x_w)) * fsg).sigmoid_() * self.att_multi, self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.func_channel(x_h) * self.func_channel(x_w) * csg), (self.func_filter(x_h) * self.func_filter(x_w) * fsg), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.dropout(self.func_channel(x_h) * self.func_channel(x_w))), (self.dropout(self.func_filter(x_h) * self.func_filter(x_w))), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # k_att = F.relu(self.func_kernel(x) - 0.8 * self.func_kernel(x_inverse))
        # k_att = k_att / (k_att.sum(dim=1, keepdim=True) + 1e-8)
        # return self.func_channel(x), self.func_filter(x), self.func_spatial(x), k_att


class KernelSpatialModulation_Local(nn.Module):
    """
    KernelSpatialModulation_Local: KSM 的局部通道分支。
    论文中提到，KSM 的局部通道分支使用轻量级 1-D 卷积捕获局部通道信息，
    并预测一个密集调制矩阵，用于每个滤波器元素的精细调制。
    这个模块还结合了傅里叶变换，可能用于处理通道的频率信息。
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        """
        初始化 KernelSpatialModulation_Local 模块。

        Args:
            channel (int, optional): 输入特征图的通道数。
            kernel_num (int): 并行卷积核的数量（FDConv 中的 'n'）。
            out_n (int): 用于调制每个输入通道的维度数量。
                         根据论文关于 KSM 预测 alpha 维度为 k × k × Cin × Cout，
                         以及此模块的输出 `(B, kernel_num, Cin, k1*k2)` 来看，
                         这里的 `out_n` 最可能是 `kernel_size * kernel_size`，
                         即每个输入通道的每个核都会生成一个 `k x k` 的空间调制。
            k_size (int): 用于 Conv1d 的自适应核大小。如果 `channel` 参数被提供，
                          则会根据 `channel` 的大小计算一个自适应的 `k_size`。
            use_global (bool): 是否使用全局信息（这里具体指的是利用傅里叶变换处理通道信息）。
        """
        super(KernelSpatialModulation_Local, self).__init__() # 调用父类 nn.Module 的构造函数
        self.kn = kernel_num # 保存并行核数量
        self.out_n = out_n # 保存输出调制维度
        self.channel = channel # 保存通道数
        # 如果提供了通道数，则根据通道数自适应计算 Conv1d 的核大小
        # 这里的计算方式是 round((log2(channel) / 2) + 0.5)，然后确保结果是奇数。
        if channel is not None: k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        # 1D 卷积层，输入通道为 1，输出通道为 `kernel_num * out_n`。
        # 这个 1D 卷积是对通道维度进行操作，类似于 Squeeze-and-Excitation (SE) 网络中的操作。
        # 输入 x 的形状通常是 (B, 1, C) 或 (B, 2, C)（如果考虑 x_std）。
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6) # 初始化卷积权重为一个很小的值
        self.use_global = use_global # 保存是否使用傅里叶变换的标志
        if self.use_global:
            # 如果使用傅里叶变换，则创建一个可学习的复数权重，用于调制傅里叶谱。
            # `complex_weight` 的形状是 `(1, self.channel // 2 + 1, 2)`，
            # 其中 `self.channel // 2 + 1` 是实数 FFT 后的频率维度大小，`2` 代表实部和虚部。
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-6)
            # self.norm = nn.GroupNorm(num_groups=32, num_channels=channel) # 被注释掉的组归一化
        self.norm = nn.LayerNorm(self.channel) # 层归一化，对通道维度进行归一化
        # self.norm_std = nn.LayerNorm(self.channel) # 被注释掉的额外层归一化
        # trunc_normal_(self.complex_weight, std=.02) # 被注释掉的权重初始化方法
        # self.sigmoid = nn.Sigmoid() # 被注释掉的激活函数
        # nn.init.constant(self.conv.weight.data) # nn.init.normal_(self.conv.weight, std=1e-6) # 被注释掉的权重初始化
        # nn.init.zeros_(self.conv.weight) # 被注释掉的权重初始化

    def forward(self, x, x_std=None):
        """
        前向传播函数。
        Args:
            x (Tensor): 输入特征。预期形状为 (B, C, 1, 1) 或者经过处理后的 (B, 1, C)。
                        代码中 `x.squeeze(-1).transpose(-1, -2)` 会将其转换为 (B, 1, C) 的形状。
            x_std (Tensor, optional): 未使用的参数，可能用于处理特征的标准差信息。
        Returns:
            Tensor: 局部调制矩阵的对数值 (logits)。
                    形状为 `(B, kernel_num, Cin, out_n)`。
                    其中 `out_n` 应该是 `kernel_size * kernel_size`，
                    表示为每个输入通道的每个并行核生成的空间调制值。
        """
        # 将输入 x 调整为 (B, 1, C) 的形状，以便进行 1D 卷积
        # 例如：如果输入是 (B, C, 1)，squeeze(-1) 变为 (B, C)，transpose(-1, -2) 变为 (B, 1, C)。
        x = x.squeeze(-1).transpose(-1, -2) # b,c,1, -> b,c, -> b,1,c,
        b, _, c = x.shape # 获取 batch_size, 第二个维度（通常为1），通道数 c

        if self.use_global: # 如果启用傅里叶变换处理通道信息
            # 对输入 x 进行实数 FFT (rfft)，得到复数谱。
            # `x_rfft` 的形状为 `(B, 1 or 2, C // 2 + 1)`，其中 `C // 2 + 1` 是频率维度大小。
            x_rfft = torch.fft.rfft(x.float(), dim=-1) # b, 1 or 2, c // 2 +1
            # print(x_rfft.shape) # 调试打印形状
            # 将可学习的复数权重 `self.complex_weight` 应用于 FFT 结果的实部和虚部。
            # `self.complex_weight[..., 0][None]` 提取实部权重并增加 batch 维度。
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            # 将调制后的实部和虚部重新组合成复数，再进行逆实数 FFT (irfft)。
            # 然后将结果加回到原始 x，这是一种在频域进行通道信息注入的方式。
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)),
                                    dim=-1) # b, 1, c // 2 +1

        x = self.norm(x) # 对处理后的 x 进行层归一化
        # x = torch.stack([self.norm(x[:, 0]), self.norm_std(x[:, 1])], dim=1) # 被注释掉的额外归一化
        # 应用 1D 卷积，生成注意力对数值 (logits)。
        # 卷积后的形状是 `(B, kernel_num * out_n, C)`。
        att_logit = self.conv(x)
        # print(att_logit.shape) # 调试打印形状
        # print(att.shape) # 调试打印形状
        # 将 `att_logit` reshape 为 `(B, kernel_num, out_n, C)`。
        # 再转置维度为 `(B, kernel_num, C, out_n)`。
        # 例如，如果 `out_n = kernel_size * kernel_size`，则为 `(B, kernel_num, Cin, K*K)`。
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c) # b, kn, k1*k2, cin
        att_logit = att_logit.permute(0, 1, 3, 2) # b, kn, cin, k1*k2
        # print(att_logit.shape) # 调试打印形状
        return att_logit # 返回局部调制对数值


# --- FrequencyBandModulation (FBM) 模块 ---
class FrequencyBandModulation(nn.Module):
    """
    FrequencyBandModulation (FBM) 模块。
    对应论文中的“频带调制（Frequency Band Modulation）”部分。
    FBM 将卷积核分解为多个频带，并应用空间特定的调制，
    自适应地调整每个频率分量在不同空间位置上的影响。
    """

    def __init__(self,
                 in_channels,  # 输入通道数
                 k_list=[2, 4, 8],  # 频率分解的尺度列表，例如 [2, 4, 8] 对应 1/2, 1/4, 1/8 频率
                 lowfreq_att=False,  # 是否对最低频率部分应用注意力调制
                 fs_feat='feat',  # 用于生成注意力的特征来源，'feat'表示使用输入特征
                 act='sigmoid',  # 注意力激活函数类型
                 spatial='conv',  # 空间调制的方式，'conv'表示使用卷积
                 spatial_group=1,  # 空间调制卷积的组数
                 spatial_kernel=3,  # 空间调制卷积的核大小
                 init='zero',  # 空间调制卷积的权重初始化方式
                 **kwargs,  # 其他未使用的关键字参数
                 ):
        super().__init__()  # 调用 nn.Module 父类的构造函数
        # k_list.sort() # 被注释掉的行，表示可能考虑过对 k_list 排序
        # print() # 调试打印
        self.k_list = k_list  # 保存频率尺度列表
        # self.freq_list = freq_list # 被注释掉的行
        self.lp_list = nn.ModuleList()  # 低通滤波器列表，此处未实际使用，但在概念上可能代表不同频带
        self.freq_weight_conv_list = nn.ModuleList()  # 频带权重（空间调制）卷积列表
        self.fs_feat = fs_feat  # 保存特征来源
        self.in_channels = in_channels  # 保存输入通道数
        # self.residual = residual # 被注释掉的残差连接标志
        # 如果 spatial_group 大于 64，则将其设置为输入通道数，这是一种深度可分离卷积的常用设置
        if spatial_group > 64: spatial_group = in_channels
        self.spatial_group = spatial_group  # 保存空间调制卷积的组数
        self.lowfreq_att = lowfreq_att  # 保存是否对最低频率应用注意力的标志
        if spatial == 'conv':  # 如果空间调制方式是卷积
            self.freq_weight_conv_list = nn.ModuleList()  # 初始化频带权重卷积列表
            _n = len(k_list)  # 频带数量
            if lowfreq_att:  _n += 1  # 如果对最低频率也应用注意力，则频带数量加1
            for i in range(_n):  # 为每个频带创建空间调制卷积
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,  # 输入通道为特征图的通道数
                                             out_channels=self.spatial_group,  # 输出通道为空间调制组数
                                             stride=1,  # 步长为1
                                             kernel_size=spatial_kernel,  # 卷积核大小
                                             groups=self.spatial_group,  # 组卷积，实现深度可分离特性
                                             padding=spatial_kernel // 2,  # 保持空间维度不变
                                             bias=True)  # 包含偏置
                if init == 'zero':  # 如果初始化策略是 'zero'
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)  # 权重用小标准差正态分布初始化
                    freq_weight_conv.bias.data.zero_()  # 偏置初始化为0
                else:
                    # raise NotImplementedError # 如果是其他初始化方式，可能未实现
                    pass
                self.freq_weight_conv_list.append(freq_weight_conv)  # 将创建的卷积层添加到列表中
        else:  # 如果空间调制方式不是 'conv'
            raise NotImplementedError  # 抛出未实现错误
        self.act = act  # 保存注意力激活函数类型

    def sp_act(self, freq_weight):
        """
        对频带权重应用指定的激活函数。
        Args:
            freq_weight (Tensor): 频带权重（卷积输出）。
        Returns:
            Tensor: 经过激活函数处理后的频带权重。
        """
        if self.act == 'sigmoid':  # 如果激活函数是 sigmoid
            freq_weight = freq_weight.sigmoid() * 2  # sigmoid 后乘以2，使其范围从 (0,1) 变为 (0,2)
        elif self.act == 'tanh':  # 如果激活函数是 tanh
            freq_weight = 1 + freq_weight.tanh()  # tanh 后加1，使其范围从 (-1,1) 变为 (0,2)
        elif self.act == 'softmax':  # 如果激活函数是 softmax
            # softmax 后乘以通道数，可能是为了在加权求和时保持能量
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError  # 抛出未实现错误
        return freq_weight

    def forward(self, x, att_feat=None):
        """
        前向传播函数。
        Args:
            x (Tensor): 输入特征图。
            att_feat (Tensor, optional): 用于生成注意力的特征。如果为 None，则使用 x。
        Returns:
            Tensor: 经过频带调制后的输出特征图。
        """
        if att_feat is None: att_feat = x  # 如果没有指定注意力特征，则使用输入特征 x
        x_list = []  # 用于存储每个频带调制后的特征
        x = x.to(torch.float32)  # 将输入特征转换为 float32 类型
        pre_x = x.clone()  # 复制一份 x 作为前一个频带的输入（用于计算高频部分）
        b, _, h, w = x.shape  # 获取 batch_size, 通道数, 高, 宽
        h, w = int(h), int(w)  # 将高宽转换为整数

        # 计算输入特征的实数 FFT（rfft2），并进行正交归一化
        x_fft = torch.fft.rfft2(x, norm='ortho')

        # 遍历每个频率尺度，进行频带分离和调制
        for idx, freq in enumerate(self.k_list):
            # 创建一个与 x_fft 形状相同的全零掩码 (B, 1, H, W//2+1)
            mask = torch.zeros_like(x_fft[:, 0:1, :, :], device=x.device)
            # 获取 FFT 空间中的频率索引
            _, freq_indices = get_fft2freq(d1=x.size(-2), d2=x.size(-1), use_rfft=True)
            # mask[:,:,round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0 # 被注释掉的中心区域掩码
            # print(freq_indices.shape) # 调试打印
            # 获取频率索引的最大值，用于确定频带边界
            freq_indices = freq_indices.max(dim=-1, keepdims=False)[0]
            # print(freq_indices) # 调试打印
            # 根据频率 `freq` 设置掩码，`freq_indices < 0.5 / freq` 对应于低频部分
            # 例如，如果 freq=2，则 `0.5/2 = 0.25`，表示频率指数小于0.25的部分为低频
            mask[:, :, freq_indices < 0.5 / freq] = 1.0  # 将对应低频部分的掩码设为1
            # print(mask.sum()) # 调试打印掩码的和

            # 将 FFT 谱 `x_fft` 与掩码相乘，然后进行逆实数 FFT (irfft2) 得到低频部分
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), dim=(-2, -1), norm='ortho')
            try:  # 尝试获取实部，可能为了兼容旧版PyTorch或特定数据类型
                low_part = low_part.real
            except:
                pass
            high_part = pre_x - low_part  # 高频部分 = 前一个输入 - 当前低频部分
            pre_x = low_part  # 更新 pre_x 为当前的低频部分，用于下一个迭代（下一个频带的高频计算）

            # 使用空间调制卷积生成频带权重
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)  # 应用激活函数

            # 将频带权重与高频部分相乘
            # 将 freq_weight 和 high_part reshape 成 (b, spatial_group, -1, h, w)
            # 这里的 -1 代表 in_channels / spatial_group
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h,
                                                                                           w)
            x_list.append(tmp.reshape(b, -1, h, w))  # 将调制后的高频部分重新 reshape 并添加到列表中

        if self.lowfreq_att:  # 如果对最低频率部分也应用注意力调制
            freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)  # 使用列表中的下一个卷积层
            freq_weight = self.sp_act(freq_weight)  # 应用激活函数
            # 将频带权重与最终剩余的（最低频）部分 `pre_x` 相乘
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))  # 添加到列表中
        else:
            x_list.append(pre_x)  # 否则，直接将最低频部分添加到列表中（不进行调制）

        x = sum(x_list)  # 将所有频带调制后的特征求和
        return x  # 返回最终的特征图


def get_fft2freq(d1, d2, use_rfft=False):
    """
    计算二维FFT空间中的频率坐标和它们距离原点的L2范数，并进行排序。
    可以用于生成频域掩码。

    Args:
        d1 (int): 第一个维度（行）的大小。
        d2 (int): 第二个维度（列）的大小。
        use_rfft (bool): 是否使用实数FFT（rfft），会影响第二个维度的频率范围。
    Returns:
        Tuple[Tensor, Tensor]:
            - sorted_coords (Tensor): 排序后的频率坐标，形状为 (2, freq_elements)。
            - freq_hw (Tensor): 原始的2D频率坐标网格，形状为 (d1, d2, 2)。
    """
    # 计算行和列的频率分量
    freq_h = torch.fft.fftfreq(d1)  # 第一个维度 (d1) 的频率，范围通常是 [-0.5, 0.5)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)  # 如果使用 rfft，第二个维度 (d2) 的频率，范围通常是 [0, 0.5]
    else:
        freq_w = torch.fft.fftfreq(d2)  # 否则，第二个维度的频率，范围通常是 [-0.5, 0.5)

    # 使用 meshgrid 创建一个 2D 频率坐标网格
    # freq_hw 的形状为 (d1, d2, 2)，其中最后一个维度表示 (freq_h_val, freq_w_val)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    # print(freq_hw) # 调试打印
    # print(freq_hw.shape) # 调试打印
    # 计算频率空间中每个点到原点 (0, 0) 的距离（L2 范数）
    dist = torch.norm(freq_hw, dim=-1)  # dist 的形状为 (d1, d2)
    # print(dist.shape) # 调试打印
    # 展平距离张量并排序，同时获取排序后的索引
    sorted_dist, indices = torch.sort(dist.view(-1))  # Flatten the distance tensor for sorting
    # print(sorted_dist.shape) # 调试打印

    # 根据排序后的扁平索引，获取对应的 2D 频率坐标
    if use_rfft:
        d2_effective = d2 // 2 + 1  # 如果使用 rfft，第二个维度在FFT后会变为 d2 // 2 + 1
    else:
        d2_effective = d2  # 否则等于原始 d2
    # 将扁平索引 `indices` 转换回 2D 坐标 (row_idx, col_idx)
    sorted_coords = torch.stack([indices // d2_effective, indices % d2_effective],
                                dim=-1)  # Convert flat indices to 2D coords
    # print(sorted_coords.shape) # 调试打印
    # # Print sorted distances and corresponding coordinates # 调试打印循环
    # for i in range(sorted_dist.shape[0]):
    #     print(f"Distance: {sorted_dist[i]:.4f}, Coordinates: ({sorted_coords[i, 0]}, {sorted_coords[i, 1]})")

    if False:  # 调试绘图部分，此处被禁用
        # Plot the distance matrix as a grayscale image
        plt.imshow(dist.cpu().numpy(), cmap='gray', origin='lower')
        plt.colorbar()
        plt.title('Frequency Domain Distance')
        plt.show()
    # 返回转置后的排序坐标（2行，freq_elements列）和原始频率网格
    return sorted_coords.permute(1, 0), freq_hw


# @CONV_LAYERS.register_module() # for mmdet, mmseg # 被注释掉的行，可能用于MMDetection或MMSegmentation框架的模块注册
class FDConv(nn.Conv2d):
    """
    Frequency Dynamic Convolution (FDConv) 模块。
    这是论文中提出的核心卷积层，它集成了傅里叶不相交权重 (FDW)、
    核空间调制 (KSM) 和频带调制 (FBM)。
    它继承自 PyTorch 的 nn.Conv2d，并在此基础上增加了动态和频率相关的特性。
    """

    def __init__(self,
                 *args,
                 # 接受 nn.Conv2d 的所有标准参数，如 in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
                 reduction=0.0625,  # KSM 全局分支的通道缩减率
                 kernel_num=4,  # FDW 中并行卷积核的数量
                 use_fdconv_if_c_gt=16,  # 只有当通道数（in_channels 或 out_channels 中较小者）大于等于此值时才使用 FDConv 特性
                 use_fdconv_if_k_in=[1, 3],  # 只有当 kernel_size 在此列表（例如 [1, 3]）中时才使用 FDConv 特性
                 use_fbm_if_k_in=[3],  # 只有当 kernel_size 在此列表（例如 [3]）中时才使用 FBM 特性
                 kernel_temp=1.0,  # 核注意力（FDW）的温度参数
                 temp=None,  # KSM 全局分支的其他注意力的温度参数，如果为 None 则使用 kernel_temp
                 att_multi=2.0,  # KSM 全局分支注意力的乘法因子
                 param_ratio=1,  # FDW 中傅里叶域参数的复制比例，例如 2 意味着参数量不变但多样性增加
                 param_reduction=0.5,  # FDW 中傅里叶域参数的实际缩减率，用于控制傅里叶系数的数量
                 ksm_only_kernel_att=False,  # KSM 全局分支是否只输出核注意力
                 att_grid=1,  # KSM 全局分支中未使用的参数
                 use_ksm_local=True,  # 是否使用 KSM 局部分支
                 ksm_local_act='sigmoid',  # KSM 局部分支的激活函数
                 ksm_global_act='sigmoid',  # KSM 全局分支的激活函数
                 spatial_freq_decompose=False,  # KSM 全局分支是否进行空间频率分解
                 convert_param=True,  # 是否将原始 nn.Conv2d 的权重转换为傅里叶域参数 (dft_weight)
                 linear_mode=False,  # 是否处于线性模式（可能用于 1x1 卷积或全连接层）
                 fbm_cfg={  # FBM 模块的配置字典
                     'k_list': [2, 4, 8],
                     'lowfreq_att': False,
                     'fs_feat': 'feat',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                     'spatial_kernel': 3,
                     'init': 'zero',
                     'global_selection': False,  # 未使用的参数
                 },
                 **kwargs,  # 传递给 nn.Conv2d 父类的其他参数
                 ):
        super().__init__(*args, **kwargs)  # 调用 nn.Conv2d 父类的构造函数，初始化标准卷积层属性
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt  # 保存通道数阈值
        self.use_fdconv_if_k_in = use_fdconv_if_k_in  # 保存核大小列表
        self.kernel_num = kernel_num  # 保存并行核数量
        self.param_ratio = param_ratio  # 保存参数复制比例
        self.param_reduction = param_reduction  # 保存参数缩减率
        self.use_ksm_local = use_ksm_local  # 保存是否使用 KSM 局部分支
        self.att_multi = att_multi  # 保存注意力乘法因子
        self.spatial_freq_decompose = spatial_freq_decompose  # 保存空间频率分解标志
        self.use_fbm_if_k_in = use_fbm_if_k_in  # 保存 FBM 生效的核大小列表

        self.ksm_local_act = ksm_local_act  # 保存 KSM 局部激活函数
        self.ksm_global_act = ksm_global_act  # 保存 KSM 全局激活函数
        assert self.ksm_local_act in ['sigmoid', 'tanh']  # 确保 KSM 局部激活函数类型有效
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']  # 确保 KSM 全局激活函数类型有效

        ### Kernel num & Kernel temp setting (核数量和核温度设置)
        if self.kernel_num is None:  # 如果没有指定核数量
            self.kernel_num = self.out_channels // 2  # 默认设置为输出通道数的一半
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)  # 计算核温度
        if temp is None:  # 如果没有指定 KSM 全局温度
            temp = kernel_temp  # 则使用核温度

        # print('*** kernel_num:', self.kernel_num)  # 调试打印核数量
        # 计算 alpha 值，可能与傅里叶域参数的缩放有关
        self.alpha = min(self.out_channels,
                         self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction

        # 如果不满足使用 FDConv 的条件（通道数或核大小不符合要求），则直接返回，保持为标准卷积层
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return  # 此时该 FDConv 实例将行为与 nn.Conv2d 完全一致

        # 初始化 KSM 全局分支
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0],
                                                         groups=self.groups,  # 传递卷积的组数
                                                         temp=temp,  # KSM 全局温度
                                                         kernel_temp=kernel_temp,  # 核注意力温度
                                                         reduction=reduction,  # 通道缩减率
                                                         kernel_num=self.kernel_num * self.param_ratio,
                                                         # KSM 全局分支要预测的核数量（考虑 param_ratio）
                                                         kernel_att_init=None,  # 核注意力初始化（由 KSM_Global 内部处理）
                                                         att_multi=att_multi,  # 注意力乘法因子
                                                         ksm_only_kernel_att=ksm_only_kernel_att,  # 是否只输出核注意力
                                                         act_type=self.ksm_global_act,  # KSM 全局激活函数类型
                                                         att_grid=att_grid,  # 未使用
                                                         stride=self.stride,  # 卷积步长
                                                         spatial_freq_decompose=spatial_freq_decompose)  # 是否进行空间频率分解

        # 如果核大小在 FBM 生效的列表中，则初始化 FBM 模块
        if self.kernel_size[0] in use_fbm_if_k_in:
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)  # 传入 FBM 配置
            # self.FBM = OctaveFrequencyAttention(2 * self.in_channels // 16, **fbm_cfg) # 被注释掉的备选FBM实现
            # self.channel_comp = ChannelPool(reduction=16) # 被注释掉的通道压缩模块

        # 如果启用 KSM 局部分支，则初始化 KSM_Local 模块
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]))  # out_n 可能是 Cout * K * K

        self.linear_mode = linear_mode  # 保存线性模式标志
        # 调用 convert2dftweight 方法进行权重转换和傅里叶域参数的设置
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        """
        将原始卷积核权重转换为傅里叶域的表示（傅里叶不相交权重 FDW）。
        Args:
            convert_param (bool): 是否将原始权重转换为 DFT 权重并删除原始权重。
        """
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        # 获取傅里叶频率索引，用于后续将傅里叶系数分配到不同的核
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)  # 形状为 (2, 总频率元素数)
        # freq_indices = freq_indices.reshape(2, self.kernel_num, -1) # 被注释掉的重塑

        # 将原始卷积权重 (Cout, Cin, K, K) 重新排列并合并维度，以方便进行 2D FFT
        # (out_channels, in_channels, kernel_h, kernel_w) -> (out_channels, kernel_h, in_channels, kernel_w) -> (out_channels*kernel_h, in_channels*kernel_w)
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        # 对合并后的权重进行 2D 实数 FFT (rfft2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))  # 形状为 (d1*k1, d2*k2 // 2 + 1)

        # 参数缩减逻辑
        if self.param_reduction < 1:  # 如果 param_reduction 小于 1，表示要减少实际存储的傅里叶系数数量
            # 随机打乱频率索引，然后只取前 `param_reduction` 比例的索引
            # 目的是只保留一部分傅里叶系数，实现参数量的缩减
            freq_indices = freq_indices[:, torch.randperm(freq_indices.size(1), generator=torch.Generator().manual_seed(
                freq_indices.size(1)))]  # 2, indices
            freq_indices = freq_indices[:, :int(freq_indices.size(1) * self.param_reduction)]  # 2, indices
            # 将复数傅里叶系数分解为实部和虚部，并堆叠
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            # 根据选定的频率索引，提取对应的傅里叶系数
            # weight_rfft[freq_indices[0, :], freq_indices[1, :]] 筛选出指定索引的系数
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            # 重新 reshape，并根据 param_ratio 进行复制，同时进行归一化
            # `(min(self.out_channels, self.in_channels) // 2)` 可能是一个归一化因子
            weight_rfft = weight_rfft.reshape(-1, 2)[None,].repeat(self.param_ratio, 1, 1) / (
                    min(self.out_channels, self.in_channels) // 2)
        else:  # 如果 param_reduction 不小于 1，表示不进行傅里叶系数的缩减，而是使用所有系数
            # 将所有复数傅里叶系数分解为实部和虚部，并堆叠
            # 然后根据 param_ratio 进行复制，同时进行归一化
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1,
                                                                                                  1, 1) / (
                                  min(self.out_channels, self.in_channels) // 2)  # param_ratio, d1*k1, d2*k2//2+1, 2

        if convert_param:  # 如果 `convert_param` 为 True，表示要将原始权重转换为 DFT 权重
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)  # 将傅里叶域参数注册为可学习参数
            del self.weight  # 删除原始的卷积权重，因为现在模型将使用 dft_weight
        else:  # 如果 `convert_param` 为 False，表示不转换
            if self.linear_mode:  # 如果是线性模式
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)  # 可能对权重进行 squeeze 操作

        self.indices = []  # 用于存储每个 `param_ratio` 副本对应的频率索引
        for i in range(self.param_ratio):
            # 将 `freq_indices` 重塑为 `(2, kernel_num, -1)`，以便每个并行核获得不相交的频率索引子集
            # 这里的 -1 表示每个核分到的频率元素数量
            self.indices.append(freq_indices.reshape(2, self.kernel_num,
                                                     -1))  # param_ratio, 2, kernel_num, d1 * k1 * (d2 * k2 // 2 + 1) // kernel_num

    def get_FDW(self, ):
        """
        在 forward 过程中如果 `dft_weight` 不存在（即 `convert_param` 为 False），
        则动态计算傅里叶不相交权重 (FDW)。
        """
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        # 重新整形原始权重以便进行 FFT
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        # 计算 2D 实数 FFT
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))  # d1 * k1, d2 * k2 // 2 + 1
        # 将复数傅里叶系数分解为实部和虚部，堆叠，并根据 param_ratio 复制，进行归一化
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1, 1,
                                                                                              1) / (
                              min(self.out_channels, self.in_channels) // 2)  # param_ratio, d1, d2, k*k, 2
        return weight_rfft

    def forward(self, x):
        """
        FDConv 的前向传播函数。
        Args:
            x (Tensor): 输入特征图。
        Returns:
            Tensor: 经过 FDConv 处理后的输出特征图。
        """
        # 如果不满足使用 FDConv 的条件，则退化为标准 nn.Conv2d
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return super().forward(x)  # 调用父类 nn.Conv2d 的 forward 方法

        # 计算 KSM 全局分支的输入（全局平均池化）
        global_x = F.adaptive_avg_pool2d(x, 1)
        # 通过 KSM 全局分支获取各种注意力权重
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)

        if self.use_ksm_local:  # 如果使用 KSM 局部分支
            # global_x_std = torch.std(x, dim=(-1, -2), keepdim=True) # 被注释掉的计算标准差
            # 通过 KSM 局部分支获取高分辨率注意力对数值
            hr_att_logit = self.KSM_Local(global_x)  # b, kn, cin, cout * ratio, k1*k2,
            # 将 hr_att_logit reshape 到 (B, 1, Cin, Cout, K, K) 的形状
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0],
                                                self.kernel_size[1])
            # hr_att_logit = hr_att_logit + self.hr_cin_bias[None, None, :, None, None, None] + self.hr_cout_bias[None, None, None, :, None, None] + self.hr_spatial_bias[None, None, None, None, :, :] # 被注释掉的偏置添加
            # 调整维度顺序为 (B, 1, Cout, Cin, K, K) 以便与卷积核维度对齐
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':  # 根据激活函数类型计算最终的 hr_att
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1  # 如果不使用 KSM 局部分支，则 hr_att 为1

        b = x.size(0)  # 获取 batch_size
        batch_size, in_planes, height, width = x.size()  # 获取输入特征图的详细尺寸
        # 初始化一个用于构建 DFT 权重图的零张量
        DFT_map = torch.zeros(
            (b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2),
            device=x.device)  # 形状对应于傅里叶域中的 (B, H_filter, W_filter_half+1, 2)

        # 重塑核注意力，以便与 FDW 的参数比例和并行核数量对齐
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)

        # 获取傅里叶域权重 (dft_weight)
        if hasattr(self, 'dft_weight'):  # 如果 dft_weight 已经存在（在 init 中转换过）
            dft_weight = self.dft_weight
        else:  # 否则，动态计算 FDW
            dft_weight = self.get_FDW()

        # 根据核注意力组合傅里叶域权重
        for i in range(self.param_ratio):  # 遍历每个参数比例副本
            indices = self.indices[i]  # 获取该副本对应的频率索引
            if self.param_reduction < 1:  # 如果进行了参数缩减
                # 针对缩减后的傅里叶系数进行处理
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]  # 重塑 dft_weight
                # 将每个核的傅里叶系数与相应的核注意力相乘，并加到 DFT_map 中
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:  # 如果没有进行参数缩减
                # 针对完整的傅里叶系数进行处理
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha  # 1, kernel_num, -1, 2
                # print(w.shape) # 调试打印
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)

        # 将组合后的傅里叶域权重通过逆实数 FFT (irfft2) 转换回空间域
        # 得到自适应卷积核 (adaptive_weights)
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2)).reshape(batch_size, 1,
                                                                                                self.out_channels,
                                                                                                self.kernel_size[0],
                                                                                                self.in_channels,
                                                                                                self.kernel_size[1])
        # 调整自适应卷积核的维度顺序为 (B, 1, Cout, Cin, K_H, K_W)
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)  # 应该是 (B, 1, Cout, Cin, K_H, K_W)

        # print(spatial_attention, channel_attention, filter_attention) # 调试打印注意力

        if hasattr(self, 'FBM'):  # 如果 FBM 模块存在
            x = self.FBM(x)  # 将输入特征 x 通过 FBM 模块进行频带调制
            # x = self.FBM(x, self.channel_comp(x)) # 被注释掉的 FBM 备选调用方式

        # 判断进行何种卷积聚合方式：
        # 如果卷积核的总参数量小于输入/输出特征图尺寸之和乘以高宽（这是一种估算计算量的方式）
        # 通常意味着卷积核参数相对较少，适合将注意力直接乘以卷积核
        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (
                in_planes + self.out_channels) * height * width:
            # print(channel_attention.shape, filter_attention.shape, hr_att.shape) # 调试打印注意力形状
            # 聚合所有注意力权重：空间注意力 * 通道注意力 * 滤波器注意力 * 自适应权重 * 高分辨率注意力
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            # aggregate_weight = spatial_attention * channel_attention * adaptive_weights * hr_att # 被注释掉的备选聚合方式
            aggregate_weight = torch.sum(aggregate_weight, dim=1)  # 对并行核维度求和，得到最终的聚合权重
            # print(aggregate_weight.abs().max()) # 调试打印聚合权重的最大绝对值
            # 将聚合权重重塑为 PyTorch F.conv2d 所需的形状：(Batch_size*Groups*Cout, Cin/Groups, K, K)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)  # 将输入特征重塑为 (1, B*Cin, H, W)
            # 使用 F.conv2d 进行卷积操作
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups * batch_size)  # groups 设为 self.groups * batch_size 实现批处理卷积

            # 恢复输出特征图的形状
            if isinstance(filter_attention, float):  # 如果 filter_attention 是 float 类型（即没有被计算，是 skip 出来的1.0）
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:  # 如果 filter_attention 是 Tensor 类型
                output = output.view(batch_size, self.out_channels, output.size(-2),
                                     output.size(-1))  # * filter_attention.reshape(b, -1, 1, 1) # 被注释掉的滤波器注意力应用方式
        else:  # 另一种卷积聚合方式，可能适用于卷积核参数相对较多的情况
            aggregate_weight = spatial_attention * adaptive_weights * hr_att  # 不包含 channel_attention 和 filter_attention
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float):  # 如果 channel_attention 不是 float 类型
                x = x * channel_attention.view(b, -1, 1, 1)  # 将 channel_attention 应用到输入特征 x 上
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            # if isinstance(filter_attention, torch.FloatTensor): # 被注释掉的类型检查
            if isinstance(filter_attention, float):
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2),
                                     output.size(-1)) * filter_attention.view(b, -1, 1,
                                                                              1)  # 将 filter_attention 应用到输出特征图上

        if self.bias is not None:  # 如果存在偏置
            output = output + self.bias.view(1, -1, 1, 1)  # 添加偏置
        return output  # 返回最终输出

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ):
        """
        用于模块的性能分析（计算参数量和FLOPs）。
        TODO: to edit it 表示此处仍需完善。
        """
        b_sz, c, h, w = input.shape  # 获取输入张量的批大小、通道数、高、宽
        seq_len = h * w  # 序列长度（空间像素数量）

        # FFT iFFT (傅里叶变换和逆傅里叶变换) 的计算量估算
        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c  # 估算 FFT/iFFT 的 MACs (乘加操作数)
        # others (其他操作的参数和MACs)
        # params = macs = sum([p.numel() for p in self.parameters()]) # 被注释掉的计算所有参数的MACs
        # 这里的 macs 估算可能与自注意力的计算相关，而不是直接的卷积参数
        params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
        # // 2 min n become half after fft # 注释：FFT后，最小的n会减半
        macs = macs * b_sz * seq_len  # 将估算的MACs乘以批大小和序列长度

        # return input, params, macs # 被注释掉的返回方式
        return input, params, macs + m_ff  # 返回输入、参数量和总MACs (包括FFT/iFFT)


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleConvNet, self).__init__()

        # 第一个标准卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # 第二个卷积层，我们在这里使用 FDConv
        # 注意：FDConv 的参数需要根据你的需求进行调整
        # 确保 in_channels 和 out_channels 满足 FDConv 内部的条件 (e.g., >= 16)
        # 确保 kernel_size 满足 FDConv 内部的条件 (e.g., in [1, 3])
        self.fdconv1 = FDConv(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            kernel_num=64,  # 较大的 kernel_num 以体现 FDConv 优势
            use_fdconv_if_c_gt=16,  # 确保通道数满足条件
            use_fdconv_if_k_in=[1, 3],  # 确保核大小满足条件
            use_fbm_if_k_in=[3],  # 启用 FBM
            param_reduction=0.5,  # 尝试减少傅里叶参数量
            use_ksm_local=True,  # 启用 KSM 局部
            ksm_global_act='sigmoid'  # KSM 全局注意力使用 softmax
        )
        self.bn_fd1 = nn.BatchNorm2d(128)
        self.relu_fd1 = nn.ReLU()

        # 第三个标准卷积层
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # 缩小空间维度
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        # 调用 FDConv 层
        x = self.relu_fd1(self.bn_fd1(self.fdconv1(x)))

        x = self.relu2(self.bn2(self.conv2(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平操作
        x = self.fc(x)
        return x


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型实例
    model = SimpleConvNet(in_channels=3, num_classes=10).to(device)

    # 打印模型结构，确认 FDConv 已被正确实例化
    print(model)

    # 打印模型参数总量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    # 创建一个模拟输入
    # 注意：FLOPs计算需要一个具体的输入尺寸
    dummy_input = torch.randn(2, 3, 32, 32).to(device)  # batch size usually 1 for FLOPs calculation

    # 使用 thop.profile 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)  # verbose=False 减少详细输出
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # 进行前向传播
    print("\nPerforming forward pass...")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # 检查 FDConv 是否按照预期工作
    # 如果 FDConv 的条件不满足，它会退化为 nn.Conv2d
    # 我们可以通过其参数数量来大致判断
    if hasattr(model.fdconv1, 'dft_weight'):
        print("\nFDConv is active and uses DFT weights!")
        print(f"FDConv DFT weight shape: {model.fdconv1.dft_weight.shape}")
    else:
        print("\nFDConv is NOT active (fell back to nn.Conv2d) due to init conditions.")
        print("Check `use_fdconv_if_c_gt` and `use_fdconv_if_k_in` parameters.")

    # 模拟训练步骤
    print("\nSimulating a training step...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dummy_labels = torch.randint(0, 10, (4,)).to(device)  # 随机生成标签

    optimizer.zero_grad()
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()
    print(f"Loss after one step: {loss.item():.4f}")

    print("\nFDConv model demonstration complete.")
