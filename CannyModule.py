import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision.utils import save_image


class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DWConvNobr(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConvNobr, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class LearnableSmoothing(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(LearnableSmoothing, self).__init__()
        self.log_sigma_sq = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.channel_adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_adapter = nn.Identity()

    def _get_gaussian_kernel(self, sigma, input_channels_for_kernel):
        device = sigma.device
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=device)
        yy, xx = torch.meshgrid(ax, ax, indexing='ij')

        sigma_safe = torch.clamp(sigma, min=0.1)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma_safe ** 2))
        kernel = kernel / torch.sum(kernel)
        gaussian_kernel_base = kernel.unsqueeze(0).unsqueeze(0)

        if self.out_channels == input_channels_for_kernel:
            final_kernel = gaussian_kernel_base.repeat(self.out_channels, 1, 1, 1)
        elif self.out_channels == 1:
            final_kernel = gaussian_kernel_base.repeat(1, input_channels_for_kernel, 1, 1)
        else:
            raise ValueError(
                f"Unsupported out_channels ({self.out_channels}) for LearnableSmoothing with dynamic kernel and input_channels_for_kernel ({input_channels_for_kernel}).")

        return final_kernel

    def forward(self, x):
        x_adapted = self.channel_adapter(x)
        current_input_channels = x_adapted.shape[1]
        sigma = torch.exp(0.5 * self.log_sigma_sq)
        gaussian_kernel = self._get_gaussian_kernel(sigma, current_input_channels)
        groups = current_input_channels if self.out_channels == current_input_channels else 1
        return F.conv2d(x_adapted, gaussian_kernel, padding=self.kernel_size // 2, groups=groups)

class LearnableGradient(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(LearnableGradient, self).__init__()
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=kernel_size // 2, bias=False)
        self.conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=kernel_size // 2, bias=False)
        self._init_sobel_weights(in_channels, out_channels)

    def _init_sobel_weights(self, in_channels, out_channels):
        sobel_x_np = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=np.float32)
        sobel_y_np = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=np.float32)
        sobel_x = torch.from_numpy(sobel_x_np).float().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.from_numpy(sobel_y_np).float().unsqueeze(0).unsqueeze(0)
        self.conv_x.weight.data.copy_(sobel_x.repeat(out_channels, in_channels, 1, 1))
        self.conv_y.weight.data.copy_(sobel_y.repeat(out_channels, in_channels, 1, 1))

    def forward(self, x):
        Gx = self.conv_x(x)
        Gy = self.conv_y(x)
        magnitude = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-6)
        direction = torch.atan2(Gy, Gx)
        return magnitude, direction

class LearnableNMS(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=3):
        super(LearnableNMS, self).__init__()

        self.local_suppressor = nn.Sequential(
            DWConv(in_channels * 3, 64),  # 修改输入通道为 in_channels * 3
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, magnitude, direction):
        if magnitude.shape[1] != self.in_channels or direction.shape[1] != self.in_channels:
            raise ValueError(
                f"Input magnitude/direction channels mismatch. Expected {self.in_channels}, got {magnitude.shape[1]} and {direction.shape[1]}")

        sin_direction = torch.sin(direction)
        cos_direction = torch.cos(direction)

        # 将幅值、正弦方向、余弦方向拼接
        combined_features = torch.cat([magnitude, sin_direction, cos_direction], dim=1)  # [B, Cin*3, H, W]
        # =======================================================

        suppression_factor = self.local_suppressor(combined_features)

        if self.out_channels == 1 and magnitude.shape[1] > 1:
            nms_output = magnitude * suppression_factor.repeat(1, magnitude.shape[1], 1, 1)
        elif self.out_channels == magnitude.shape[1]:
            nms_output = magnitude * suppression_factor
        else:
            raise ValueError("Unsupported suppression_factor output channels vs magnitude channels.")

        return nms_output

class LearnableHysteresis(nn.Module):
    def __init__(self, in_channels, out_channels=1):  # in_channels 是 NMS 输出的通道数
        super(LearnableHysteresis, self).__init__()
        self.high_threshold = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.low_threshold = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.log_steepness = nn.Parameter(torch.tensor(np.log(50.0), dtype=torch.float32))

        # 输入通道现在是 (in_channels + in_channels + in_channels) = in_channels * 3
        self.in_channels_linker = in_channels * 3
        self.edge_linker = nn.Sequential(
            DWConv(self.in_channels_linker, 64),
            DWConv(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1))

        # 最终的 Sigmoid 激活在 forward 中显式调用
        self.sigmoid = nn.Sigmoid()

    def forward(self, nms_output):
        # 确保 high_t 的上下限都是 Tensor 并与参数在同一设备
        high_t = torch.clamp(self.high_threshold,
                             torch.tensor(0.01, device=self.high_threshold.device),
                             torch.tensor(0.99, device=self.high_threshold.device))

        # 确保 low_t 的上下限都是 Tensor 并与参数在同一设备
        low_t = torch.clamp(self.low_threshold,
                            torch.tensor(0.001, device=self.low_threshold.device),
                            high_t - torch.tensor(0.001, device=self.high_threshold.device))
        steepness = torch.exp(self.log_steepness)

        # 对 NMS 输出的每个通道独立进行阈值处理
        strong_edges_prob = torch.sigmoid((nms_output - high_t) * steepness)
        weak_edges_prob = torch.sigmoid((nms_output - low_t) * steepness) - strong_edges_prob
        weak_edges_prob = torch.clamp(weak_edges_prob, 0, 1)

        # 拼接输入通道 (NMS_output, strong_prob, weak_prob)
        input_for_linker = torch.cat([nms_output, strong_edges_prob, weak_edges_prob], dim=1)

        # 边缘连接网络 (现在是简单的卷积序列)
        final_edge_map_logits = self.edge_linker(input_for_linker)
        final_edge_map = self.sigmoid(final_edge_map_logits)

        return final_edge_map

class LearnableCannyEdgeModule(nn.Module):
    def __init__(self, decoder_feature_channels, output_edge_channels=1):
        super(LearnableCannyEdgeModule, self).__init__()
        self.smoothing = LearnableSmoothing(in_channels=decoder_feature_channels,
                                            out_channels=decoder_feature_channels)
        self.gradient = LearnableGradient(in_channels=decoder_feature_channels,
                                          out_channels=decoder_feature_channels, kernel_size=3)
        self.nms = LearnableNMS(in_channels=decoder_feature_channels,
                                out_channels=decoder_feature_channels)
        self.hysteresis = LearnableHysteresis(in_channels=decoder_feature_channels,
                                              out_channels=output_edge_channels)


    def forward(self, decoder_features):
        x_smooth = self.smoothing(decoder_features)  # 平滑后的特征
        magnitude, direction = self.gradient(x_smooth)  # 梯度幅值和方向 (都是 decoder_feature_channels 通道)
        nms_output = self.nms(magnitude, direction)  # NMS 后的特征 (decoder_feature_channels 通道)
        # final_edge_map, high_t, low_t, _ = self.hysteresis(nms_output)  # 最终边缘图 (1通道), 强弱边缘概率 (decoder_feature_channels 通道)
        final_edge_map = self.hysteresis(nms_output)  # 最终边缘图 (1通道), 强弱边缘概率 (decoder_feature_channels 通道)
        return final_edge_map


