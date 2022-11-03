import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# anchors = [
#     [1,1, 1,0.5, 0.5,1],
#     [2,2, 2,1.5, 1.5,2],
#     [0.5,0.5, 0.5,0.25, 0.25,0.5]
# ]
anchors = [[85.0, 84.0], [47.0, 41.0], [104.0, 48.0], [56.0, 111.0], [111.0, 104.0], [53.0, 79.0], [87.0, 115.0], [74.0, 49.0], [113.0, 75.0]]
# anchors = [[148.0, 98.0], [174.0, 230.0], [224.0, 147.0], [111.0, 222.0], [106.0, 158.0], [94.0, 83.0], [223.0, 204.0], [169.0, 166.0], [208.0, 95.0]]

# class Model(nn.Module):

#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.num_anchors = len(anchors) * len(anchors[0]) // 2
#         self.num_out_per_anchor = args.num_classes + 5
#         self.backbone = resnet50(num_classes=self.num_out_per_anchor * self.num_anchors)
#         a = torch.tensor(anchors).float().view(1, -1, 2) / 64
#         self.register_buffer('anchors', a)


#     def forward(self, x):
#         x = self.backbone(x).view(x.shape[0], self.num_anchors, self.num_out_per_anchor).contiguous()
#         if not self.training:
#             x = x.sigmoid()
#             xy = x[..., 0:2]
#             wh = x[..., 2:4] ** 2 * self.anchors
#             y = torch.cat((xy, wh, x[..., 5:]), dim=-1)
#             select_idx = torch.argmax(x[..., 4], dim=1)
#             select_output = y[range(y.shape[0]), select_idx]
#         return x if self.training else select_output

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_anchors = len(anchors) * len(anchors[0]) // 2
        self.num_out_per_anchor = args.num_classes + 5
        self.backbone = resnet50(num_classes=self.num_out_per_anchor * self.num_anchors)
        self.grid = torch.zeros(1)
        
        sp = 256
        with torch.no_grad():
            stride = sp / self.forward(torch.zeros(1, 3, sp, sp)).shape[-2]
        a = torch.tensor(anchors).float().view(-1, 2)
        self.stride = stride
        self.register_buffer('anchors', a / stride)
        self.register_buffer('anchor_grid', a.clone().view(1, -1, 1, 1, 2))


    def forward(self, x):
        x = self.backbone(x)
        batchsize, _, ny, nx = x.shape
        x = x.view(batchsize, self.num_anchors, self.num_out_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            xy = (y[..., 0:2] * 2. - 0.5 + self.grid) * self.stride / self.args.img_size  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid.view(1, self.num_anchors, 1, 1, 2) / self.args.img_size  # wh
            y = torch.cat((xy, wh, x[..., 4:]), dim=-1).view(batchsize, -1, self.num_out_per_anchor)
            y_out = torch.cat((xy, wh, x[..., 5:]), dim=-1).view(batchsize, -1, self.num_out_per_anchor-1)
            select_idx = torch.argmax(y[..., 4], dim=1)
            select_output = y_out[range(y.shape[0]), select_idx]

        return x if self.training else select_output

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# class BackBone(nn.Module):

#     def __init__(self):
#         super(BackBone).__init__()
#         self.module = resnet50()
    
#     def forward(self, x):
#         out = self.module(x)
#         return out

# class RegressHeader(nn.module):
    
#     def __init__(self):
#         super(RegressHeader).__init__()
#         self.conv1 = 
        

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out


class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out


class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, bias=False)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
 
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
        Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
        Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model