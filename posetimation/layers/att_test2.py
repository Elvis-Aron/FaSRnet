import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels_x1,
                 in_channels_x2,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=True,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels_x1 = in_channels_x1
        self.in_channels_x2 = in_channels_x2
        self.inter_channels = inter_channels


        if self.inter_channels is None:
            self.inter_channels = in_channels_x1 // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels_x1,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels_x1,
                        kernel_size=1,
                        stride=1,
                        padding=0),
                bn(self.in_channels_x1)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels_x1,
                             kernel_size=1,
                             stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        # 对x1 处理
        self.theta = conv_nd(in_channels=self.in_channels_x1,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1, padding=0)
        # 对x2 处理
        self.phi = conv_nd(in_channels=self.in_channels_x2,
                           out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x1, x2):
        '''
        :param x: (b, c, t, h, w)
        :return:
        计算x2 和x1 的相关性
        然后给x1加权
        '''

        batch_size = x1.size(0)
        # 需要进行加权的特征
        g_x = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # 本身的特征
        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x1 = theta_x1.permute(0, 2, 1)
        # 额外引入的特征

        phi_x2 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x1, phi_x2)
        f_div_C = F.softmax(f, dim=-1)
        # 给当前特征加权
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        W_y = self.W(y)
        z = W_y + x1

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels_x1,
                 in_channels_x2,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):

        super(NONLocalBlock1D, self).__init__(in_channels_x1,
                                              in_channels_x2,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self,  in_channels_x1,
                 in_channels_x2,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels_x1,
                                              in_channels_x2,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self,  in_channels_x1,
                 in_channels_x2,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels_x1,
                                              in_channels_x2,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img1 = torch.zeros(2, 3, 20)
        img2 = torch.zeros(2, 2, 20)
        net = NONLocalBlock1D(in_channels_x1=3,
                             in_channels_x2=2,
                              inter_channels=2,
                              sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img1,img2)
        print(out.size())

        img1 = torch.zeros(2, 3, 20, 20)
        img2 = torch.zeros(2, 2, 20, 20)
        net = NONLocalBlock2D(in_channels_x1=3,
                             in_channels_x2=2,
                              inter_channels=2,
                              sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img1,img2)
        print(out.size())

        img1 = torch.zeros(2, 3, 8, 20, 20)
        img2 = torch.zeros(2, 2, 8,20, 20)
        net = NONLocalBlock3D(in_channels_x1=3,
                             in_channels_x2=2,
                              inter_channels=2,
                              sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img1,img2)
        print(out.size())


