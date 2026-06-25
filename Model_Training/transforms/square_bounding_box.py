from typing import Tuple

import torch


class SquareBoundingBox(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        img, bounding_box = img

        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        deltas = torch.zeros_like(bounding_box, device=img.device)
        deltas[0] = torch.where(height > width, (height - width) // 2, 0)
        deltas[1] = torch.where(width > height, (width - height) // 2, 0)
        deltas[2] = torch.where(height > width, (height - width - deltas[0]), 0)
        deltas[3] = torch.where(width > height, (width - height - deltas[1]), 0)
        deltas[0] = -deltas[0]
        deltas[1] = -deltas[1]

        bounding_box += deltas

        return img, bounding_box

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "()"
        return format_string
