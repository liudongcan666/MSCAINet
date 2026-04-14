import torch
import torch.nn as nn
import torch.nn.functional as F
# https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html 在这个网站找到与你环境相匹配的mmcv按照命令
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from einops import rearrange
import warnings
import numbers
from timm.models.layers import DropPath
# pip install -U openmim
# mim install mmcv==2.0.0

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SE_Block(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Adaptive_Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., reduction=4):
        super().__init__()
        self.dwconv0 = Conv(dim, dim, 7, g=dim, act=False)
        self.dwconv1 = Conv(dim, dim, 5, g=dim, act=False)
        self.se = SE_Block(dim, reduction)  # Add SE module for channel attention
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.hidden_weight = nn.Parameter(torch.ones(dim, 1, 1))  # learnable weight for high-dimensional control

    def forward(self, x):
        input = x
        x = self.dwconv0(x) + self.dwconv1(x) + x
        x = self.se(x)  # Apply SE block
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = x * self.hidden_weight  # Apply learnable weight for implicit high-dimensional control
        x = input + self.drop_path(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class SEWithAdvancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改进后的通道和空间注意力结合的SE模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(SEWithAdvancedSpatialAttention, self).__init__()

        # 通道注意力部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid_channel = nn.Sigmoid()  # Sigmoid 激活生成通道权重

        # 空间注意力部分 (改进)
        self.conv3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)  # 3x3卷积
        self.conv5x5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, bias=False)  # 5x5卷积
        self.conv7x7 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)  # 将多个尺度的空间信息进行融合
        self.sigmoid_spatial = nn.Sigmoid()  # Sigmoid 激活生成空间权重

    def forward(self, x):
        # 通道注意力部分
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y_channel = self.sigmoid_channel(y)  # 通道权重
        x_channel = x * y_channel  # 按通道加权

        # 空间注意力部分（多尺度卷积）
        out_3x3 = self.conv3x3(x_channel)  # 使用3x3卷积提取空间特征
        out_5x5 = self.conv5x5(x_channel)  # 使用5x5卷积提取空间特征
        out_7x7 = self.conv7x7(x_channel)  # 使用7x7卷积提取空间特征

        # 融合多个尺度的空间特征
        out_fused = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)  # 拼接不同尺度的特征
        out_fused = self.conv_fuse(out_fused)  # 使用1x1卷积融合空间特征

        y_spatial = self.sigmoid_spatial(out_fused)  # 生成空间权重
        return x_channel * y_spatial  # 同时加权通道和空间
# spatial-spectral domain attention learning(SDL)
class SDL_attention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, mode="ch*sp"):
        super(SDL_attention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.mode = mode

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()
        self.se = SEWithAdvancedSpatialAttention(in_channels=inplanes, reduction=16)
        # new
        # self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
        #                        bias=True)
        # todo conv改成kernel_size=1或5试试

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    # HR spatial attention
    def spatial_attention(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)

        mask_ch = self.sigmoid(context)

        return mask_ch

    # HR spectral attention
    def spectral_attention(self, x):

        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()

        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)

        mask_sp = self.sigmoid(context)

        return mask_sp

    def forward(self, x):
        mask_ch = self.spatial_attention(x)
        mask_sp = self.spectral_attention(x)
        se = self.se(x)
        if self.mode == "ch":
            out = x * mask_ch
        elif self.mode == "sp":
            out = x * mask_sp
        elif self.mode == "ch+sp":
            out = x * mask_ch + x * mask_sp
        elif self.mode == "ch*sp":
            out = x * mask_ch * mask_sp * se
        else:
            raise ValueError("DDA mode is unsupported")

        return out + x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm(dim, LayerNorm_type) #如果看不懂上面的几个bias函数,或者觉得很繁琐,可以把这一行替换为普通的layerNorm函数,上面的直接删除就可以了。
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.dda1 = SDL_attention(256,256)
        self.dda2 = SDL_attention(256, 256)

    def forward(self, x):
        b, c, h, w = x.shape # (B,C2,H/4,W/4)
        x1 = self.norm1(x) # (B,C2,H/4,W/4)-->(B,C2,H/4,W/4)

        # 分别在x轴和y轴上提取多尺度特征
        attn_00 = self.conv0_1(x1) # x轴上的1×7conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_01 = self.conv0_2(x1) # y轴上的7×1conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_10 = self.conv1_1(x1) # x轴上的1×11conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_11 = self.conv1_2(x1) # y轴上的11×1conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_20 = self.conv2_1(x1) # x轴上的1×21conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_21 = self.conv2_2(x1) # y轴上的21×1conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)

        # 分别融合x轴和y轴上的多尺度特征
        out1 = attn_00 + attn_10 + attn_20 # 将x轴上的三个尺度的卷积后的特征进行相加：(B,C2,H/4,W/4) = (B,C2,H/4,W/4) + (B,C2,H/4,W/4) + (B,C2,H/4,W/4)
        out2 = attn_01 + attn_11 + attn_21 # 将y轴上的三个尺度的卷积后的特征进行相加：(B,C2,H/4,W/4) = (B,C2,H/4,W/4) + (B,C2,H/4,W/4) + (B,C2,H/4,W/4)
        out1 = self.project_out(out1) # 对x轴相加的特征通过1×1Conv进行融合：(B,C2,H/4,W/4)
        out2 = self.project_out(out2) # 对y轴相加的特征通过1×1Conv进行融合,这里x轴和y轴共享一个1×1conv层：(B,C2,H/4,W/4)

        out3 = self.dda1(out1)
        out4 = self.dda2(out2)

        # # 分别通过x轴和y轴上的融合后的多尺度特征，来生成qkv
        # k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4)) C2=k*d, k是注意力头的个数，d是每个注意力头的通道数
        # v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4))
        # k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        # v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        # q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        # q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4))
        # q1 = torch.nn.functional.normalize(q1, dim=-1)
        # q2 = torch.nn.functional.normalize(q2, dim=-1)
        # k1 = torch.nn.functional.normalize(k1, dim=-1)
        # k2 = torch.nn.functional.normalize(k2, dim=-1)
        #
        # # 交叉轴注意力：q来自y轴的特征,k/v来自x轴特征
        # attn1 = (q1 @ k1.transpose(-2, -1)) # 计算注意力矩阵：(B,k,H/4,d*(W/4)) @ (B,k,d*(W/4),H/4) = (B,k,H/4,H/4)
        # attn1 = attn1.softmax(dim=-1) # softmax归一化注意力矩阵：(B,k,H/4,H/4)-->(B,k,H/4,H/4)
        # out3 = (attn1 @ v1) + q1 # 对v矩阵进行加权：(B,k,H/4,H/4) @ (B,k,H/4,d*(W/4)) = (B,k,H/4,d*(W/4))
        #
        # # 交叉轴注意力：q来自x轴的特征,k/v来自y轴特征
        # attn2 = (q2 @ k2.transpose(-2, -1)) # 计算注意力矩阵：(B,k,W/4,d*(H/4)) @ (B,k,d*(H/4),W/4) = (B,k,W/4,W/4)
        # attn2 = attn2.softmax(dim=-1) #  softmax归一化注意力矩阵：(B,k,W/4,W/4)-->(B,k,W/4,W/4)
        # out4 = (attn2 @ v2) + q2 # 对v矩阵进行加权：(B,k,W/4,W/4) @ (B,k,W/4,d*(H/4)) = (B,k,W/4,d*(H/4))

        # # 将x和y轴上的注意力输出变换为与输入相同的shape
        # out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(B,k,H/4,d*(W/4))-->(B,C2,H/4,W/4)
        # out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(B,k,W/4,d*(H/4))-->(B,C2,H/4,W/4)

        # 对x和y轴注意力的输出进行融合,并添加残差连接，shape保持不变
        out = self.project_out(out3) + self.project_out(out4) + x

        return out


class MCAHead(nn.Module):
    def __init__(self, in_channels, image_size, heads, c1_channels,
                 **kwargs):
        super().__init__()
        self.image_size = image_size
        self.decoder_level = Attention(in_channels[1], heads, LayerNorm_type='WithBias')
        self.conv_cfg = None
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.align = ConvModule(
            in_channels[3],
            in_channels[0],
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.squeeze = ConvModule(
            sum((in_channels[1], in_channels[2], in_channels[3])),
            in_channels[1],
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels[1] + in_channels[0],
                in_channels[3],
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                in_channels[3],
                in_channels[3],
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))


    def forward(self, inputs):
        """Forward function."""
        # inputs = self._transform_inputs(inputs)
        inputs = [resize(
            level,
            size=(self.image_size,int(self.image_size/2)),
            mode='bilinear',
            align_corners=False
        ) for level in inputs] #将X2/X3/X4进行上采样,与X1保持相同的特征图大小,但是通道没有变化
        y1 = torch.cat([inputs[1], inputs[2], inputs[3]], dim=1) # 将X2/X3/X4进行拼接：(B,C2+C3+C4,H/4,W/4)
        x = self.squeeze(y1) # 将拼接后的多个特征进行降维,和X2保持相同的通道数量：(B,C2+C3+C4,H/4,W/4)-->(B,C2,H/4,W/4)
        x = self.decoder_level(x) # 执行交叉轴注意力：(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        x = torch.cat([x, inputs[0]], dim=1) # 将交叉轴注意力的输出与X1进行拼接,这是为了能够更好的利用X1的边界信息,因为X1的特征图尺寸最大：(B,C2,H/4,W/4)-concat-(B,C1,H/4,W/4) == (B,C1+C2,H/4,W/4)
        x = self.sep_bottleneck(x) # 两层卷积：(B,C1+C2,H/4,W/4)-->(B,C4,H/4,W/4)-->(B,C4,H/4,W/4)
        output = self.align(x) # 恢复和X1相同的shape: (B,C4,H/4,W/4)-->(B,C1,H/4,W/4)


        return output


class SAFL(nn.Module):
    def __init__(self, dim=2048, part_num=6) -> None:
        super().__init__()

        self.part_num = part_num
        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, 2048)))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim)))

        self.active = nn.Sigmoid()

        self.abc = MCAHead(in_channels=[2048,256,512,1024], image_size=18, heads=8, c1_channels=48)
        self.act = nn.ReLU6()

        self.s1 = Adaptive_Star_Block(256)
        self.s2 = Adaptive_Star_Block(512)
        self.s3 = Adaptive_Star_Block(1024)
        self.s4 = Adaptive_Star_Block(2048)


    def forward(self, x1,x2,x3,x4):
        B, C, H, W = x4.shape  # [80, 2048, 18, 9]

        x1 = self.s1(x1)
        x2 = self.s2(x2)
        x3 = self.s3(x3)
        x4 = self.s4(x4)

        input = [x4,x1,x2,x3]
        x = x4 + self.act(self.abc(input))

        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C] , [80, 162, 2048]
        x_pos = x + self.pos_embeding  # [80, 162, 2048] = [80, 162, 2048] + [162, 2048]
        attn = self.part_tokens @ x_pos.transpose(-1, -2)  # [80, 7, 162] = [7, 2048] @ [80, 2048, 162]
        attn = self.active(attn)  # [80, 7, 162]
        x = attn @ x / H / W  # [80, 7, 2048] = [80, 7, 162] @ [80, 162, 2048]


        # [80, 14336] , [80, 7, 162]
        return x.view(B, -1), attn