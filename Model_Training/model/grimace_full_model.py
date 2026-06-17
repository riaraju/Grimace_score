from typing import List, Tuple

import timm
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights


class GrimaceFullModel(torch.nn.Module):
    def __init__(self, base_model_name: str, input_size: Tuple[int, int], num_channels: int = 3, hidden_layer_sizes: List[int] = None, dropouts: List[float] = None):
        super(GrimaceFullModel, self).__init__()
        if base_model_name == 'vit_b_16':
            self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.base_model.heads = torch.nn.Identity()
        elif base_model_name == 'vit_b_16_swag':
            self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.base_model.heads = torch.nn.Identity()
        elif base_model_name == 'vit_b_32':
            self.base_model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            self.base_model = timm.create_model(base_model_name, in_chans=num_channels, pretrained=True, num_classes=0, global_pool='avg')

        base_model_output = self.base_model(torch.zeros(1, num_channels, input_size[0], input_size[1]))
        num_features = base_model_output.shape[1]

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [128]

        if dropouts is None:
            dropouts = [0.0] * len(hidden_layer_sizes)

        input_size = num_features

        self.hidden_layers = torch.nn.Sequential()

        for hidden_layer_size, dropout in zip(hidden_layer_sizes, dropouts):
            self.hidden_layers.append(torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, hidden_layer_size),
                torch.nn.ReLU(),
            ))
            input_size = hidden_layer_size

        self.au_heads = torch.nn.ModuleList()

        if len(hidden_layer_sizes) == 0 and len(dropouts) == 1:
            for _ in range(5):
                self.au_heads.append(torch.nn.Sequential(
                    torch.nn.Dropout(dropouts[0]),
                    torch.nn.Linear(input_size, 4)
                ))
        else:
            for _ in range(5):
                self.au_heads.append(torch.nn.Sequential(
                    torch.nn.Linear(input_size, 4)
                ))

    def forward(self, x):
        out = self.base_model(x)
        out = self.hidden_layers(out)
        out = [head(out) for head in self.au_heads]
        out = torch.concat(out, dim=1)
        return out
