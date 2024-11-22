import torch
import torch.nn as nn
from torchvision.models import (efficientnet_b0, efficientnet_b2, efficientnet_b4, efficientnet_v2_s, mnasnet1_0,
                                mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, 
                                shufflenet_v2_x1_5, googlenet)

class BB_Guesser_Model(nn.Module):
    def __init__(self, config, backbone_name=None, proposals=2, angles=1, resize_size=224, orient_neurons=128, dim_neurons=256):
        super(BB_Guesser_Model, self).__init__()
        self.classification = config['Models']['3d_guesser_classification']
        if(self.classification):
            self.sincos = config['Models']['3d_guesser_proposals']
            self.proposals=proposals
        else:
            self.sincos = 1
            self.proposals = proposals
            
        
        self.resize_size = resize_size

        # Map backbone names to model constructors and feature extraction points
        backbone_dict = {
            'efficientnet_b0': efficientnet_b0,
            'MNASNet1_0': mnasnet1_0,
            'MobileNet_V2': mobilenet_v2,
            'MobileNet_V3_Small_Weights': mobilenet_v3_small,
            'ShuffleNet_V2_X1_5': shufflenet_v2_x1_5,
            'efficientnet_b2': efficientnet_b2,
            'efficientnet_b4': efficientnet_b4,
            'GoogLeNet': googlenet,
            'MobileNet_V3_Large_Weights': mobilenet_v3_large,
            'efficientnet_v2_S_Weights': efficientnet_v2_s,
        }

        if backbone_name in backbone_dict:
            backbone = backbone_dict[backbone_name](weights='IMAGENET1K_V1')
            self.backbone_out = self.get_backbone_output_features(backbone)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.img_out_size = self.calc_backbone_out()
        else:
            self.backbone_out = 3
            self.backbone = None
            self.img_out_size = 1

        if self.backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.flatten = nn.Flatten()
        self.orientation = nn.Sequential(
            nn.Linear(self.backbone_out * self.img_out_size * self.img_out_size, orient_neurons),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(orient_neurons, orient_neurons),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(orient_neurons, proposals * self.sincos)
        )
        self.dimension = nn.Sequential(
            nn.Linear(self.backbone_out * self.img_out_size * self.img_out_size, dim_neurons),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(dim_neurons, dim_neurons),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(dim_neurons, 3)
        )

    def get_backbone_output_features(self, backbone):
        """Extract the number of output features from the classifier layer, if it exists."""
        if hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Sequential):
            for layer in (backbone.classifier):
              if isinstance(layer, nn.Linear):
                return (layer.in_features)
              if isinstance(layer, nn.Conv2d):
                return (layer.in_channels)
            raise ValueError(f"No suitable final layer found in the classifier for backbone {type(backbone).__name__}")
        elif hasattr(backbone, 'fc'):
            return backbone.fc.in_features
        elif hasattr(backbone, 'features') and isinstance(backbone.features, nn.Sequential):
            return backbone.features[-1].num_features
        else:
            raise ValueError(f"Cannot determine output feature size for backbone {type(backbone).__name__}")

    def calc_backbone_out(self):
        test = torch.randn(1, 3, self.resize_size, self.resize_size)
        with torch.no_grad():
            output_test = self.backbone(test)
        output_size = output_test.shape[-1]
        return output_size

    def forward(self, x):
        if self.backbone:
            out_x = self.backbone(x)
        else:
            out_x = x

        out_x_2 = self.flatten(out_x)
        orientation = self.orientation(out_x_2)
        if(self.classification):
            orientation = orientation.view(-1, self.sincos, self.proposals)
        dimension = self.dimension(out_x_2)
        return orientation, dimension
