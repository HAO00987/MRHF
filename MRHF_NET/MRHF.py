import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel
from timm.data.transforms_factory import create_transform
from PIL import Image
import os
import torch.nn.functional as F

class BTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BTM, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv_3x3_d3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, dilation=3, padding=3)
        self.conv_3x3_d5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, dilation=5, padding=5)
        self.conv_3x3_d7 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, dilation=7, padding=7)
        self.conv_1x1 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
    def forward(self, x):
        conv_3x3_out = self.conv_3x3(x)
        conv_3x3_d3_out = self.conv_3x3_d3(x)
        conv_3x3_d5_out = self.conv_3x3_d5(x)
        conv_3x3_d7_out = self.conv_3x3_d7(x)
        out = torch.cat((conv_3x3_out, conv_3x3_d3_out, conv_3x3_d5_out, conv_3x3_d7_out), dim=1)
        out = self.conv_1x1(out)
        return out

class F_X_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(F_X_Module, self).__init__()
        self.btm = BTM(in_channels//2, out_channels//2)
        self.conv_1x1 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=1)
        self.conv_1x1_down = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
    def forward(self, x):
        x = self.conv_1x1_down(x)
        btm_out = self.btm(x)
        relu_out = F.relu(btm_out)
        conv_out = self.conv_1x1(relu_out)
        return conv_out

class F_Y_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(F_Y_Module, self).__init__()
        self.btm = BTM(in_channels, out_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    def forward(self, x):
        btm_out = self.btm(x)
        relu_out = F.relu(btm_out)
        conv_out = self.conv_1x1(relu_out)
        return conv_out

class HAF(nn.Module):
    def __init__(self, in_channels):
        super(HAF, self).__init__()
        out_channels = in_channels
        self.f_x_module = F_X_Module(in_channels,out_channels)
        self.f_y_module = F_Y_Module(in_channels//2,out_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        Fx = self.f_x_module(x)
        Fy = self.f_y_module(y)
        F = Fx + Fy
        F = self.sigmoid(F)
        return F

class MambaVisionUNet(nn.Module):
    def __init__(self, encoder_model_name="MambaVision-B-1K", num_classes=1):
        super(MambaVisionUNet, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name, trust_remote_code=True)
        self.encoder_channels = [128, 256, 512, 1024]
        self.decoder1 = self._decoder_block(1024, 512)
        self.decoder2 = self._decoder_block(512, 256)
        self.decoder3 = self._decoder_block(256, 128)
        self.final_conv = nn.Conv2d(256, 64, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder4 = self._decoder_block(64, 32)
        self.final_conv2 = nn.Conv2d(32, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_avg_pool, features = self.encoder(x)
        x4 = features[3]
        x3 = features[2]
        x2 = features[1]
        x1 = features[0]
        x = self.decoder1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self._reduce_channels(x, 512)
        x = self.decoder2(x)
        x = torch.cat([x, x2], dim=1)
        x = self._reduce_channels(x, 256)
        x = self.decoder3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.final_conv(x)
        x = self.decoder4(x)
        x = self.final_conv2(x)
        out = self.upsample(x)
        return out

    def _reduce_channels(self, x, target_channels):
        in_channels = x.size(1)
        return nn.Conv2d(in_channels, target_channels, kernel_size=1).cuda()(x)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.stage1 = nn.Sequential(*list(self.model.children())[:4])
        self.stage2 = self.model.layer1
        self.stage3 = self.model.layer2
        self.stage4 = self.model.layer3
        self.stage5 = self.model.layer4

    def forward(self, x):
        stage1_out = self.stage1(x)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        stage5_out = self.stage5(stage4_out)
        return stage1_out, stage2_out, stage3_out, stage4_out, stage5_out

class MambaVisionUNet2(nn.Module):
    def __init__(self, encoder_model_name="MambaVision-B-1K", num_classes=1):
        super(MambaVisionUNet2, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name, trust_remote_code=True)
        self.a_encoder = ResNetFeatureExtractor()
        self.HAF1 = HAF(1024)
        self.HAF2 = HAF(512)
        self.HAF3 = HAF(256)
        self.HAF4 = HAF(128)
        self.encoder_channels = [128, 256, 512, 1024]
        self.decoder1 = self._decoder_block(512, 256)
        self.decoder2 = self._decoder_block(512, 128)
        self.decoder3 = self._decoder_block(256, 64)
        self.final_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder4 = self._decoder_block(64, 32)
        self.final_conv2 = nn.Conv2d(32, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_avg_pool, features = self.encoder(x)
        a_features = self.a_encoder(x)
        x4 = features[3]
        x3 = features[2]
        x2 = features[1]
        x1 = features[0]
        a_x4 = a_features[4]
        a_x3 = a_features[3]
        a_x2 = a_features[2]
        a_x1 = a_features[1]
        h1 = self.HAF1(x4, a_x4)
        h2 = self.HAF2(x3, a_x3)
        h3 = self.HAF3(x2, a_x2)
        h4 = self.HAF4(x1, a_x1)
        x = self.decoder1(h1)
        x = torch.cat([x, h2], dim=1)
        x = self.decoder2(x)
        x = torch.cat([x, h3], dim=1)
        x = self.decoder3(x)
        x = torch.cat([x, h4], dim=1)
        x = self.final_conv(x)
        x = self.decoder4(x)
        x = self.final_conv2(x)
        out = self.upsample(x)
        return out

    def _reduce_channels(self, x, target_channels):
        in_channels = x.size(1)
        return nn.Conv2d(in_channels, target_channels, kernel_size=1).cuda()(x)

def test_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaVisionUNet2(num_classes=7).to(device)
    image_path = 'VOCdevkit(damage)/VOC2007/JPEGImages/(3).jpg'
    if not os.path.exists(image_path):
        print(f"Error: The image file at {image_path} does not exist.")
        return
    image = Image.open(image_path)
    input_resolution = (3, 512, 512)
    transform = create_transform(input_size=input_resolution,
                                 is_training=False,
                                 mean=model.encoder.config.mean,
                                 std=model.encoder.config.std,
                                 crop_mode=model.encoder.config.crop_mode,
                                 crop_pct=model.encoder.config.crop_pct)
    inputs = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_unet()
