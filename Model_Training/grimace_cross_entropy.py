import torch


class GrimaceCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(GrimaceCrossEntropyLoss, self).__init__()
        self.eye_loss = torch.nn.CrossEntropyLoss()
        self.nose_loss = torch.nn.CrossEntropyLoss()
        self.cheek_loss = torch.nn.CrossEntropyLoss()
        self.ear_loss = torch.nn.CrossEntropyLoss()
        self.whisker_loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        eye_loss = self.eye_loss(outputs[:, :4], targets[:, :4])
        nose_loss = self.nose_loss(outputs[:, 4:8], targets[:, 4:8])
        cheek_loss = self.cheek_loss(outputs[:, 8:12], targets[:, 8:12])
        ear_loss = self.ear_loss(outputs[:, 12:16], targets[:, 12:16])
        whisker_loss = self.whisker_loss(outputs[:, 16:], targets[:, 16:])

        return eye_loss + nose_loss + cheek_loss + ear_loss + whisker_loss
