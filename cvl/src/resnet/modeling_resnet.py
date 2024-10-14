from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from adapters.adapter_controller_resnet_fast import AdapterController

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,  # 修改padding
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 config=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input) -> Tensor:
        x = input
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 config=None) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input) -> Tensor:
        x = input
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            config,
            num_classes: int = 10,  # 修改为CIFAR-10的10类
            clean_model=None,  # 干净模型
            adv_model=None  # 对抗模型
    ) -> None:
        super().__init__()
        self.config = config
        self.inplanes = 64
        # CIFAR-10 的输入尺寸为32x32，因此使用3x3卷积代替7x7卷积
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # CIFAR-10输入较小，不需要最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification_layer = nn.Linear(512 * block.expansion, num_classes)

        # 插入AdapterController在最后一个特征图
        if self.config.train_adapters:
            self.adapter_controller = AdapterController(self.config, 512 * block.expansion, clean_model=clean_model, adv_model=adv_model)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, config=self.config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, config=self.config))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, labels) -> Tensor:
        # ResNet的标准前向传播
        # x=x.tensor()
        #print("resnet的输入：",x)
        #print(f"x 类型: {type(x)}, x 内容: {x}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x= self.layer4(x)
        #print("resnet的最后一个特征图的shape：",x.shape)
        # 在layer4的输出后插入AdapterController
        if self.config.train_adapters and labels is not None:
            if self.training:
                x, load_loss, supervised_loss = self.adapter_controller(x, labels)
                self.config.load_loss_accm += load_loss
                self.config.supervised_loss_accm += supervised_loss
            else:
                x = self.adapter_controller(x, labels)
        #print("resnet的最后一个特征图经过AdapterController后的shape：",x.shape)
        #print("resnet的最后一个特征图经过AdapterController后的输出：",x)

        return x

    def forward(self, x: Tensor, labels) -> Tensor:
        return self._forward_impl(x, labels)

def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        config,
        **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, config, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=True)
        # 忽略 conv1 层的权重，以适应 CIFAR-10 的输入尺寸
        if 'conv1.weight' in state_dict:
            del state_dict['conv1.weight']
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet50(pretrained: bool = False, config=None, **kwargs: Any) -> ResNet:
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, config, **kwargs)

def resnet18(pretrained: bool = False, config=None, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, config, **kwargs)
