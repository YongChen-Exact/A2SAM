import torch
import torch.nn as nn
class PromptGenerator(nn.Module):
    def __init__(self, in_channels=1, out_conv_channels=1):
        super(PromptGenerator, self).__init__()
        entmap_channels = 16
        logits_channels = 16
        embedding_channels = 48
        self.out_conv_channels = out_conv_channels
        self.entmap_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=entmap_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=entmap_channels, out_channels=entmap_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.logits_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=logits_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(logits_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=logits_channels, out_channels=logits_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(logits_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_channels, out_channels=embedding_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=embedding_channels, out_channels=out_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, unc_map, coarse_logit):
        unc_map, coarse_logit = unc_map.unsqueeze(0), coarse_logit.unsqueeze(0)
        uncertainty_info = self.entmap_conv(1 - unc_map)
        pred_info = self.logits_conv(coarse_logit)
        x = torch.cat((uncertainty_info, pred_info), dim=1)
        x = self.final_conv(x)
        return x