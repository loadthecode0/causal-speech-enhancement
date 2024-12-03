"""This file implements both the non-causal and causal versions of Conv-TasNet.

The implementation is based on the paper:
"Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation"
by Yi Luo and Nima Mesgarani.

The code includes building blocks of Conv-TasNet and a wrapper to instantiate
either the causal or non-causal version based on user preference.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConvTranspose1D(nn.Module):
    """
    Implements a causal transposed convolution (deconvolution) layer.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the transposed convolution kernel.
        stride (int): Stride of the transposed convolution.
        padding (int, optional): Padding added to the input (defaults to 0).
        output_padding (int, optional): Extra padding added to the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0):
        super(CausalConvTranspose1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            output_padding=output_padding,
        )
        # Compute the required causal padding
        self.left_padding = (kernel_size - 1) - padding

    def forward(self, x):
        # Apply causal padding to ensure no future access
        x = F.pad(x, (self.left_padding, 0))
        return self.conv_transpose(x)

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super(CausalConv1D, self).__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)

class ConvBlock(torch.nn.Module):
    """Non-Causal 1D Convolutional block.

    This block uses symmetric padding, making it non-causal.

    Args:
        io_channels (int): Number of input/output channels, <B, Sc>.
        hidden_channels (int): Number of channels in the internal layers, <H>.
        kernel_size (int): Convolution kernel size of the middle layer, <P>.
        padding (int): Symmetric padding value for convolution.
        dilation (int, optional): Dilation value for convolution.
        no_residual (bool, optional): Disable residual output.
    """
    def __init__(self, io_channels: int, hidden_channels: int, kernel_size: int, 
                 dilation: int = 1, no_residual: bool = False):
        super().__init__()
        self.no_residual = no_residual
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(io_channels, hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(1, hidden_channels, eps=1e-8),
            torch.nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding= (dilation * (kernel_size - 1) // 2),
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(1, hidden_channels, eps=1e-8),
        )
        self.res_out = (
            None if no_residual else torch.nn.Conv1d(hidden_channels, io_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(hidden_channels, io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        residual = self.res_out(feature) if not self.no_residual else None
        skip_out = self.skip_out(feature)
        return residual, skip_out


class CausalConvBlock(torch.nn.Module):
    """Causal 1D Convolutional block.

    This block uses left-padding to ensure that future frames are not accessed.

    Args:
        io_channels (int): Number of input/output channels, <B, Sc>.
        hidden_channels (int): Number of channels in the internal layers, <H>.
        kernel_size (int): Convolution kernel size of the middle layer, <P>.
        dilation (int, optional): Dilation value for convolution.
        no_residual (bool, optional): Disable residual output.
    """
    def __init__(self, io_channels: int, hidden_channels: int, kernel_size: int,
                 dilation: int = 1, no_residual: bool = False):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(io_channels, hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            CausalConv1D(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
        )
        self.res_out = (
            None if no_residual else torch.nn.Conv1d(hidden_channels, io_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(hidden_channels, io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        residual = self.res_out(feature) if self.res_out else None
        skip_out = self.skip_out(feature)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    """TCN-based Mask Generator for both causal and non-causal versions.

    Args:
        causal (bool): Use causal convolutions if True, otherwise non-causal.
        Other arguments match the original Conv-TasNet implementation.
    """
    def __init__(self, input_dim: int, num_sources: int, kernel_size: int,
                 num_feats: int, num_hidden: int, num_layers: int, num_stacks: int,
                 msk_activate: str, causal: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.input_norm = torch.nn.GroupNorm(1, input_dim, eps=1e-8)
        self.input_conv = torch.nn.Conv1d(input_dim, num_feats, kernel_size=1)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        ConvBlockType = CausalConvBlock if causal else ConvBlock
        for s in range(num_stacks):
            for l in range(num_layers):
                dilation = 2**l
                self.conv_layers.append(
                    ConvBlockType(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        no_residual=(l == num_layers - 1 and s == num_stacks - 1),
                    )
                )
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * dilation
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(num_feats, input_dim * num_sources, kernel_size=1)
        self.mask_activate = torch.nn.Sigmoid() if msk_activate == "sigmoid" else torch.nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


class ConvTasNet(torch.nn.Module):
    """Conv-TasNet implementation for both causal and non-causal settings.

    Args:
        causal (bool): Use causal convolutions if True, otherwise non-causal.
        Other arguments match the original Conv-TasNet implementation.
    """
    def __init__(self, num_sources: int = 2, enc_kernel_size: int = 16,
                 enc_num_feats: int = 512, msk_kernel_size: int = 3,
                 msk_num_feats: int = 128, msk_num_hidden_feats: int = 512,
                 msk_num_layers: int = 8, msk_num_stacks: int = 3,
                 msk_activate: str = "sigmoid", causal: bool = False):
        super().__init__()
        self.encoder = CausalConv1D(
            1, enc_num_feats, kernel_size=enc_kernel_size,
            dilation=1, groups=1
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
            causal=causal,
        )
        self.decoder = CausalConvTranspose1D(
            enc_num_feats, 1, kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2, output_padding=0
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded, num_pads = self._align_num_frames_with_strides(input)
        feats = self.encoder(padded)
        masked = self.mask_generator(feats) * feats.unsqueeze(1)
        output = self.decoder(masked.view(-1, feats.size(1), feats.size(2)))
        return output.view(input.size(0), -1, input.size(2))

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        num_frames = input.size(-1)
        stride = self.encoder.stride[0]
        pad = (stride - (num_frames % stride)) % stride
        return torch.nn.functional.pad(input, (0, pad)), pad


def build_conv_tasnet(causal: bool = False, **kwargs) -> ConvTasNet:
    """Wrapper to instantiate either causal or non-causal Conv-TasNet."""
    return ConvTasNet(causal=causal, **kwargs)

'''
Usage
Instantiate Non-Causal Model:
model = build_conv_tasnet(causal=False, num_sources=2)

Instantiate Causal Model:
model = build_conv_tasnet(causal=True, num_sources=2)
'''
