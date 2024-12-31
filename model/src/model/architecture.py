import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import os

class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        out = self.bn(x)
        out = self.activation(out)
        out = self.conv(out)
        
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels=in_channels + i*growth_rate, growth_rate=growth_rate))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x
    
class TransitionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.activation(self.bn(x)))
        return self.pool(x)


class DenseNet(nn.Module):

    def __init__(self, in_channels, num_classes, growth_rate=6, num_blocks=3, num_layers_per_block=4):
        
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate*2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(growth_rate*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        in_channels = growth_rate * 2
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DenseBlock(num_layers=num_layers_per_block, in_channels=in_channels, growth_rate=growth_rate))
            in_channels = in_channels + num_layers_per_block * growth_rate
            if i != num_blocks - 1:
                self.blocks.append(TransitionLayer(in_channels=in_channels, out_channels=in_channels // 2))
                in_channels = in_channels // 2

        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.fc = nn.Linear(in_features=in_channels, out_features=num_classes)

    def forward(self, x):

        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation((self.bn(x)))
        x = F.adaptive_avg_pool2d(input=x, output_size=(1,1)).view(x.size(0), -1)

        return self.fc(x)

    def save(self, dp):
        """Save model state dict to the specified directory path."""
        save_path = os.path.join(dp, "model.pt")
        if isinstance(self, DistributedDataParallel):
            torch.save(self.module.state_dict(), save_path)
        else:
            torch.save(self.state_dict(), save_path)        


# if __name__ == "__main__":

#     x = torch.randn(size=(3,3, 128, 128))
#     denselayer = DenseLayer(in_channels=3, growth_rate=6)
#     densenet = DenseNet(in_channels=3, num_classes=10)
#     state_dict = torch.load("/home/fahad/study/kserving/rayruns/TorchTrainer_2024-12-07_13-58-08/TorchTrainer_4a35b_00000_0_2024-12-07_13-58-09/checkpoint_000099/model.pt")
#     densenet.load_state_dict(state_dict)
#     import pdb; pdb.set_trace()