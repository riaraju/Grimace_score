from typing import Tuple

import torch


class CropBoundingBox(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img, bounding_box = img

        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        black_image = torch.zeros((img.shape[0], height.item(), width.item()), dtype=img.dtype, device=img.device)

        x1_in = torch.max(torch.tensor(0), bounding_box[0])
        y1_in = torch.max(torch.tensor(0), bounding_box[1])
        x2_in = torch.min(torch.tensor(img.shape[2]), bounding_box[2])
        y2_in = torch.min(torch.tensor(img.shape[1]), bounding_box[3])

        x1_out = torch.max(torch.tensor(0), -bounding_box[0])
        y1_out = torch.max(torch.tensor(0), -bounding_box[1])
        x2_out = x1_out + x2_in - x1_in
        y2_out = y1_out + y2_in - y1_in

        black_image[:, y1_out:y2_out, x1_out:x2_out] = img[:, y1_in:y2_in, x1_in:x2_in]

        return black_image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "()"
        return format_string
