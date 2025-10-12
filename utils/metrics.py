import torch

def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label - 1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num - 1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice