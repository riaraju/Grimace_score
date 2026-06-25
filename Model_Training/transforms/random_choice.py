import random

import torch


class RandomChoice(torch.nn.Module):
    def __init__(self, transforms, weights):
        super().__init__()
        self.transforms = transforms
        self.weights = weights

    def forward(self, img):
        t = random.choices(self.transforms, weights=self.weights, k=1)[0]
        return t(img)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
