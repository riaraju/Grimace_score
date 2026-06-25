from typing import Tuple

import torch


class RandomBoundingBoxPadding(torch.nn.Module):
    def __init__(self, min_padding: float, max_padding: float):
        super().__init__()
        self.min_padding = min_padding
        self.max_padding = max_padding

    def forward(self, img: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        img, bounding_box = img

        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        padding = torch.rand(4, device=bounding_box.device) * (self.max_padding - self.min_padding) + self.min_padding
        padding *= torch.tensor([-width, -height, width, height], device=bounding_box.device)
        padding = padding.int()

        bounding_box += padding

        return img, bounding_box

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    min_padding={self.min_padding}"
        format_string += f"\n    max_padding={self.max_padding}"
        format_string += "\n)"
        return format_string
