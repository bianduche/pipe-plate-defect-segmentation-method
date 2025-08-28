# star_conv.py
# 依赖：torch >=1.7
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_star_mask(kernel_size: int, device=None, dtype=torch.float32, points=5):
    """
    生成正星形（⭐）掩码，kernel_size 假定为奇数。
    例如 kernel_size=11 的五角星掩码：
    以中心为顶点，生成对称的五角星结构
    """
    assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
    mask = torch.zeros((kernel_size, kernel_size), dtype=dtype, device=device)
    center = kernel_size // 2
    max_radius = center  # 最大半径（不超过边界）

    # 计算五角星顶点坐标
    outer_radius = max_radius * 0.8  # 外顶点半径
    inner_radius = outer_radius * 0.4  # 内顶点半径（控制五角星的凹陷程度）

    # 生成五角星的顶点坐标（极坐标转笛卡尔坐标）
    vertices = []
    for i in range(2 * points):
        # 角度：从正上方开始，每步π/points
        angle = math.pi / 2 + i * math.pi / points
        # 交替使用外半径和内半径
        radius = outer_radius if i % 2 == 0 else inner_radius
        # 计算坐标
        x = center + radius * math.cos(angle)
        y = center - radius * math.sin(angle)  # y轴向下为正
        vertices.append((int(round(x)), int(round(y))))

    # 填充五角星区域（使用扫描线算法判断点是否在多边形内）
    def point_in_polygon(x, y):
        n = len(vertices)
        inside = False
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # 遍历掩码的每个点，判断是否在五角星内
    for i in range(kernel_size):
        for j in range(kernel_size):
            if point_in_polygon(j, i):  # 注意坐标顺序 (x=j, y=i)
                mask[i, j] = 1.0

    return mask


class StarConv(nn.Module):
    """
    星形卷积模块（仅包含正星形卷积分支）
    参数：
      in_channels, out_channels: 输入/输出通道数
      kernel_size: 星形卷积核的大小（奇数）
      num_points: 星形的角数（默认5，即五角星）
      stride, padding, dilation: 传给底层 conv 的参数
      activation: 激活函数实例或 None（例如 nn.SiLU()）
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=11,  # 五角星需要较大的kernel_size才能体现形状
            num_points=5,
            stride=1,
            padding=None,
            dilation=1,
            activation=nn.SiLU()
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        if padding is None:
            padding = (kernel_size - 1) // 2  # 保持尺寸不变

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_points = num_points
        self.activation = activation

        # 星形卷积层
        self.star_conv = nn.Conv2d(
            in_channels, in_channels,  # 先保持通道数，后续通过1x1调整
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False
        )

        # 逐点卷积用于调整通道数
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # 预构建星形掩码（注册为buffer，跟随设备）
        star_mask = make_star_mask(kernel_size, dtype=torch.float32, points=num_points)
        self.register_buffer("star_mask", star_mask.unsqueeze(0).unsqueeze(0))  # [1,1,k,k]

    def forward(self, x):
        """
        x: [B, C_in, H, W]
        返回: [B, C_out, H_out, W_out]
        """
        # 应用星形掩码到卷积权重
        star_weights = self.star_conv.weight * self.star_mask  # 广播到 [C, C, k, k]

        # 执行星形卷积
        x_conv = F.conv2d(
            x,
            star_weights,
            bias=None,
            stride=self.star_conv.stride,
            padding=self.star_conv.padding,
            dilation=self.star_conv.dilation,
            groups=self.star_conv.groups
        )

        # 调整通道数并应用激活函数
        out = self.pw_conv(x_conv)
        if self.activation is not None:
            out = self.activation(out)
        return out


if __name__ == "__main__":
    # 测试模块，随机输入
    # 使用较大的kernel_size更能体现五角星形状（如11）
    net = StarConv(
        in_channels=16,
        out_channels=32,
        kernel_size=11,  # 五角星需要足够大的尺寸
        num_points=5,  # 5角星
        split_mode="split",
        fusion="concat",
        activation=nn.SiLU()
    )
    x = torch.randn(2, 16, 64, 64)  # batch=2
    y = net(x)
    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)

    # 可视化掩码（如果需要）
    mask = make_star_mask(11)
    print("星形掩码示例 (11x11):")
    print(mask.numpy().astype(int))
