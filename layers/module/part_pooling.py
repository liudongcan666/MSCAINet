import torch
import torch.nn as nn



class SAFL(nn.Module):
    def __init__(self, dim=2048, part_num=6) -> None:
        super().__init__()

        self.part_num = part_num
        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, 2048)))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim)))

        self.active = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape  # [80, 2048, 18, 9]
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C] , [80, 162, 2048]

        x_pos = x + self.pos_embeding  # [80, 162, 2048] = [80, 162, 2048] + [162, 2048]

        attn = self.part_tokens @ x_pos.transpose(-1, -2)  # [80, 7, 162] = [7, 2048] @ [80, 2048, 162]

        attn = self.active(attn)  # [80, 7, 162]

        x = attn @ x / H / W  # [80, 7, 2048] = [80, 7, 162] @ [80, 162, 2048]

        # [80, 14336] , [80, 7, 162]
        return x.view(B, -1), attn