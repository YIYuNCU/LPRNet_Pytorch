import torch
import torch.nn as nn
import torch.nn.functional as F

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out, dropout_rate=0.0):
        super(small_basic_block, self).__init__()
        self.need_proj = ch_in != ch_out
        if self.need_proj:
            self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
        )
        
    def forward(self, x):
        identity = x
        if self.need_proj:
            identity = self.proj(identity)
        out = self.block(x)
        out = F.relu(out + identity)
        return out


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        
        # Backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.block1 = small_basic_block(64, 128, dropout_rate=dropout_rate*0.5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        
        self.block2 = small_basic_block(128, 256, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        
        self.block3 = small_basic_block(256, 256, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        
        # 全局特征卷积
        self.global_conv1 = nn.Conv2d(128, 256, kernel_size=1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 256 + 256 + 256*16, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            
            nn.Conv2d(256, 512, kernel_size=(4, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate*0.5),
            
            nn.Conv2d(512, class_num, kernel_size=(1, 13)),
            nn.BatchNorm2d(class_num),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        feat1 = x
        
        # Stage 2
        x = self.block1(x)
        x = self.pool1(x)
        feat2 = x
        
        # Stage 3
        x = self.block2(x)
        x = self.pool2(x)
        feat3 = x
        
        # Stage 4
        x = self.block3(x)
        x = self.pool3(x)
        feat4 = x
        
        # 多尺度特征
        global1 = F.interpolate(self.global_conv1(F.adaptive_avg_pool2d(feat2, (1,1))),
                                size=(6,24), mode='bilinear', align_corners=False)
        global2 = F.interpolate(F.adaptive_avg_pool2d(feat3, (2,2)), size=(6,24), mode='bilinear', align_corners=False)
        global3 = F.adaptive_avg_pool2d(feat4, (4,4))  # [B, 256, 4, 4]
        
        global3 = global3.view(global3.size(0), -1, 6, 24)  # 展平到通道维度
        
        x = torch.cat([x, global1, global2, global3], dim=1)
        
        # 分类头
        x = self.classifier(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        return x


def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
    if phase == "train":
        Net.train()
    else:
        Net.eval()
    return Net