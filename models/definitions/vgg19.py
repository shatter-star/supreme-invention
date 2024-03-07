import torch
from torchvision import models

class VggModel(torch.nn.Module):
    """
    VGG model for neural style transfer.
    
    Args:
        model_type (str): Type of VGG model to use. Currently supports 'vgg19'.
        requires_grad (bool): Whether to keep model parameters trainable or not.
        show_progress (bool): Whether to show download progress when loading pretrained model.
    """

    def __init__(self, model_type='vgg19', requires_grad=False, show_progress=False):
        super().__init__()

        if model_type == 'vgg19':
            vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features
            self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
            self.content_feature_maps_index = 4 
            self.style_feature_maps_indices = list(range(len(self.layer_names)))
            self.style_feature_maps_indices.remove(4)
            self.offset = 1
        else:
            raise ValueError("Unsupported VGG model type. Currently supported: 'vgg19'.")

        self.slices = torch.nn.ModuleList()
        for x in range(1 + self.offset):
            self.slices.append(torch.nn.Sequential())
            self.slices[-1].add_module(str(x), vgg_pretrained_features[x])
        for idx, (start, end) in enumerate([(1 + self.offset, 6 + self.offset), (6 + self.offset, 11 + self.offset),
                                             (11 + self.offset, 20 + self.offset), (20 + self.offset, 22)]):
            self.slices.append(torch.nn.Sequential())
            for x in range(start, end):
                self.slices[-1].add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = []
        for slice in self.slices:
            x = slice(x)
            outputs.append(x)
        return outputs