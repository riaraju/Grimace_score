from typing import Tuple

import torch


class BoundingBoxPadding(torch.nn.Module):
    def __init__(self, padding: float):
        super().__init__()
        self.padding = padding

    def forward(self, img: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        img, bounding_box = img

        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        padding = torch.tensor([-width, -height, width, height], device=bounding_box.device) * self.padding
        padding = padding.int()

        bounding_box += padding

        return img, bounding_box

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    padding={self.padding}"
        format_string += "\n)"
        return format_string
