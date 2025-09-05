# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BasicBlock(nn.Module):
#     """Basic residual block"""
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
            
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class HRModule(nn.Module):
#     """High-Resolution Module"""
#     def __init__(self, num_branches, channels_list, num_blocks):
#         super(HRModule, self).__init__()
        
#         self.num_branches = num_branches
#         self.channels_list = channels_list
        
#         # Create branches
#         self.branches = nn.ModuleList()
#         for i in range(num_branches):
#             branch = nn.Sequential()
#             for j in range(num_blocks[i]):
#                 branch.add_module(f'block_{j}', BasicBlock(channels_list[i], channels_list[i]))
#             self.branches.append(branch)
        
#         # Fusion layer
#         self.fuse_layers = nn.ModuleList()
#         for i in range(num_branches):
#             fuse_layer = nn.ModuleList()
#             for j in range(num_branches):
#                 if i == j:
#                     fuse_layer.append(nn.Identity())
#                 elif i < j:
#                     # Downsample
#                     downsample = nn.Sequential()
#                     for k in range(j - i):
#                         if k == j - i - 1:
#                             downsample.add_module(f'conv_{k}', 
#                                 nn.Conv2d(channels_list[j], channels_list[i], 3, stride=2, padding=1, bias=False))
#                             downsample.add_module(f'bn_{k}', nn.BatchNorm2d(channels_list[i]))
#                         else:
#                             downsample.add_module(f'conv_{k}', 
#                                 nn.Conv2d(channels_list[j], channels_list[j], 3, stride=2, padding=1, bias=False))
#                             downsample.add_module(f'bn_{k}', nn.BatchNorm2d(channels_list[j]))
#                             downsample.add_module(f'relu_{k}', nn.ReLU(inplace=True))
#                     fuse_layer.append(downsample)
#                 else:
#                     # Upsample
#                     upsample = nn.Sequential(
#                         nn.Conv2d(channels_list[j], channels_list[i], 1, bias=False),
#                         nn.BatchNorm2d(channels_list[i])
#                     )
#                     fuse_layer.append(upsample)
#             self.fuse_layers.append(fuse_layer)
    
#     def forward(self, x_list):
#         # Process each branch
#         out = []
#         for i, branch in enumerate(self.branches):
#             out.append(branch(x_list[i]))
        
#         # Fusion
#         fused = []
#         for i in range(self.num_branches):
#             y = None
#             for j in range(self.num_branches):
#                 if i == j:
#                     y = out[j] if y is None else y + out[j]
#                 elif i < j:
#                     # Downsample
#                     y = self.fuse_layers[i][j](out[j]) if y is None else y + self.fuse_layers[i][j](out[j])
#                 else:
#                     # Upsample
#                     temp = self.fuse_layers[i][j](out[j])
#                     temp = F.interpolate(temp, size=out[i].shape[2:], mode='bilinear', align_corners=True)
#                     y = temp if y is None else y + temp
#             fused.append(F.relu(y))
        
#         return fused

# class HRNet(nn.Module):
#     def __init__(self, in_channels=3, num_classes=2, width=18):
#         super(HRNet, self).__init__()
        
#         # Stem
#         self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
        
#         # Stage 1
#         self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for _ in range(4)])
        
#         # Transition 1
#         self.transition1 = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(64, width, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(width),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.Conv2d(64, width*2, 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(width*2),
#                 nn.ReLU(inplace=True)
#             )
#         ])
        
#         # Stage 2
#         self.stage2 = HRModule(2, [width, width*2], [4, 4])
        
#         # Transition 2
#         self.transition2 = nn.ModuleList([
#             nn.Identity(),
#             nn.Identity(),
#             nn.Sequential(
#                 nn.Conv2d(width*2, width*4, 3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(width*4),
#                 nn.ReLU(inplace=True)
#             )
#         ])
        
#         # Stage 3
#         self.stage3 = HRModule(3, [width, width*2, width*4], [4, 4, 4])
        
#         # Final layer
#         self.final_layer = nn.Conv2d(width, num_classes, 1)
        
#     def forward(self, x):
#         # Stem
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
        
#         # Stage 1
#         x = self.layer1(x)
        
#         # Transition 1
#         x_list = []
#         for transition in self.transition1:
#             x_list.append(transition(x))
        
#         # Stage 2
#         x_list = self.stage2(x_list)
        
#         # Transition 2
#         new_x_list = []
#         for i, transition in enumerate(self.transition2):
#             if i < len(x_list):
#                 new_x_list.append(transition(x_list[i]))
#             else:
#                 new_x_list.append(transition(x_list[-1]))
#         x_list = new_x_list
        
#         # Stage 3
#         x_list = self.stage3(x_list)
        
#         # Use highest resolution output
#         x = x_list[0]
        
#         # Upsample to original size if needed
#         x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
#         # Final classification (returns logits)
#         return self.final_layer(x)