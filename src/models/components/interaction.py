import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResNet2D(nn.Module):
    def __init__(self, in_channels, num_bins, dilation_rates=[1, 2, 4, 8], num_residual_blocks=4):
        super(DilatedResNet2D, self).__init__()
        
        self.residual_blocks = nn.ModuleList([
            self._build_residual_block(in_channels, in_channels, dilation_rate) 
            for dilation_rate in dilation_rates[:num_residual_blocks]
        ])
        
        self.bin_pred = nn.Conv2d(in_channels, num_bins, kernel_size=1)

    def _build_residual_block(self, in_channels, out_channels, dilation_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x

        for block in self.residual_blocks:
            out = block(x)
            out += residual  
            out = F.relu(out)
            residual = out  

        bin_output = self.bin_pred(out)
        bin_output = F.softmax(bin_output, dim=1)

        return bin_output
