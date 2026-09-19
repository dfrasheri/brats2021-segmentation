"""3D segmentation architectures."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv-Norm-LeakyReLU twice.

    InstanceNorm rather than BatchNorm: batch size is 2 at this patch size, and
    BatchNorm statistics estimated from 2 samples are noise.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _check_divisible(x, depth):
    factor = 2 ** depth
    for name, size in zip("HWD", x.shape[2:]):
        if size % factor:
            raise ValueError(
                "input {}={} is not divisible by {} (depth={}); "
                "pad the volume or reduce depth".format(name, size, factor, depth))


class UNet3D(nn.Module):
    """3D U-Net with optional deep supervision on the coarser decoder levels."""

    def __init__(self, in_channels=4, num_classes=4, depth=4, base_channels=16,
                 deep_supervision=False):
        super().__init__()
        self.depth = depth
        self.deep_supervision = deep_supervision

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        enc_channels = []
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool3d(2))
            enc_channels.append(out_ch)
            ch = out_ch

        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            self.upsamples.append(nn.ConvTranspose3d(ch, skip_ch, 2, stride=2))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch
            use_head = deep_supervision and i in (1, 2)
            self.heads.append(nn.Conv3d(ch, num_classes, 1) if use_head else nn.Identity())
            self.heads[-1].is_head = use_head

        self.final = nn.Conv3d(ch, num_classes, 1)

    def forward(self, x):
        _check_divisible(x, self.depth)

        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        aux = []
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = up(x)
            x = dec(torch.cat([x, skips[-(i + 1)]], dim=1))
            if getattr(self.heads[i], "is_head", False):
                aux.append(self.heads[i](x))

        out = self.final(x)
        if self.deep_supervision and aux:
            return [out] + aux[::-1]
        return out


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(ch, affine=True),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


class VNet(nn.Module):
    """V-Net style: residual blocks per resolution, strided-conv downsampling."""

    def __init__(self, in_channels=4, num_classes=4, base_channels=16, depth=4):
        super().__init__()
        self.depth = depth
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.downs = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        ch = base_channels
        enc_channels = [ch]
        for _ in range(depth):
            self.downs.append(nn.Sequential(
                nn.Conv3d(ch, ch * 2, 2, stride=2, bias=False),
                nn.InstanceNorm3d(ch * 2, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ))
            ch *= 2
            self.enc_blocks.append(ResBlock(ch))
            enc_channels.append(ch)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            self.ups.append(nn.ConvTranspose3d(ch, skip_ch, 2, stride=2))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv3d(skip_ch * 2, skip_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(skip_ch, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                ResBlock(skip_ch),
            ))
            ch = skip_ch

        self.final = nn.Conv3d(ch, num_classes, 1)

    def forward(self, x):
        _check_divisible(x, self.depth)

        x = self.stem(x)
        skips = [x]
        for down, blk in zip(self.downs, self.enc_blocks):
            x = blk(down(x))
            skips.append(x)

        for i, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            x = up(x)
            x = dec(torch.cat([x, skips[-(i + 2)]], dim=1))

        return self.final(x)


def create_model(name, in_channels=4, num_classes=4, base_channels=16,
                 deep_supervision=False, device="cpu"):
    if name == "unet":
        model = UNet3D(in_channels, num_classes, depth=4,
                       base_channels=base_channels, deep_supervision=deep_supervision)
    elif name == "vnet":
        model = VNet(in_channels, num_classes, base_channels=base_channels, depth=4)
    else:
        raise ValueError("unknown model: " + str(name))
    return model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
