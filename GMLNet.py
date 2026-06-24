import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from MobileViTv3V1 import MobileViTv3_v1
from CannyModule import LearnableCannyEdgeModule
from FDConv import FDConv

class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0,groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DWConvNobr(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConvNobr, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, transform=False):
        super(BasicBlock, self).__init__()
        self.transform = transform
        self.conv1 = DWConvNobr(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DWConvNobr(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1),
            nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.transform:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, 1)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fdconv1 = nn.Sequential(
            FDConv(in_channels=64, out_channels=64, kernel_size=3, kernel_num=16, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.fdconv2 = nn.Sequential(
            FDConv(in_channels=128, out_channels=128, kernel_size=3, kernel_num=32, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.fdconv3 = nn.Sequential(
            FDConv(in_channels=256, out_channels=256, kernel_size=3, kernel_num=64, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.fdconv4 = nn.Sequential(
            FDConv(in_channels=320, out_channels=320, kernel_size=3, kernel_num=80, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True))
        self.link1 = nn.Sequential(
            Conv1x1(576, 256),
            DWConv(256, 256),
            CoordAtt(256, 256))
        self.link2 = nn.Sequential(
            Conv1x1(384, 128),
            DWConv(128, 128),
            CoordAtt(128, 128))
        self.link3 = nn.Sequential(
            Conv1x1(192, 64),
            FDConv(in_channels=64, out_channels=64, kernel_size=3, kernel_num=16, padding=1),
            CoordAtt(64, 64))
        self._init_weight()

    def forward(self, x):  # 64*128*128 128*64*64 256*32*32 320*16*16

        x1 = self.fdconv1(x[0])
        x2 = self.fdconv2(x[1])
        x3 = self.fdconv3(x[2])
        x4 = self.fdconv4(x[3])
        x4 = F.interpolate(x4, x3.shape[2:], mode='bilinear', align_corners=True)
        x34 = torch.cat((x3, x4), 1)  #576
        x34 = self.link1(x34)
        x34 = F.interpolate(x34, x2.shape[2:], mode='bilinear', align_corners=True)
        x234 = torch.cat((x2, x34), 1)
        x234 = self.link2(x234)
        x234 = F.interpolate(x234, x1.shape[2:], mode='bilinear', align_corners=True)
        x1234 = torch.cat((x1, x234), 1)
        feature = self.link3(x1234)
        return feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, FDConv):
                continue
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SGENet(nn.Module):
    def __init__(self, n_classes=7):
        super(SGENet, self).__init__()
        self.encoder = MobileViTv3_v1(image_size=(512, 512), mode='small', isThree=False)
        self.encoder.load_pretrained_model('/MobileViTv3-v1/mobilevitv3_S_voc_e50_7959/checkpoint_ema_best.pt')
        self.decoder = Decoder()
        self.EdgeOperator = LearnableCannyEdgeModule(decoder_feature_channels=64, output_edge_channels=1)
        self.SegmentHead = nn.Sequential(
            Conv1x1(65,32),
            CoordAtt(32, 32),
            nn.Conv2d(32, 1, kernel_size=1))


    def forward(self, x):
        x_size = x.size()
        fe = self.encoder(x)  # 320*16*16 256*32*32 128*64*64 64*128*128
        fd = self.decoder(fe)  # 32*256*256
        fedge = self.EdgeOperator(fd)
        input = torch.cat((fd, fedge), 1)
        output = self.SegmentHead(input)
        output = F.interpolate(output, x_size[2:], mode='bilinear', align_corners=True)
        return output

if __name__ == '__main__':

    test_data = torch.rand(2, 3, 128, 128).cuda()
    model = SGENet()
    model = model.cuda()
    preds = model(test_data)
    print('Success')




