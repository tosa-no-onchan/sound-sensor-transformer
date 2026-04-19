
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F

class VideoAutoEncoder(nn.Module):
    def __init__(self, num_frames=8): # 16に変更
        super(VideoAutoEncoder, self).__init__()
        # エンコーダ（ResNet + Transformer）
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Transformerの定義
        #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=3)

        # デコーダ（512次元のベクトルから 224x224 の画像を復元する簡易的な構成）
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224 * 224),
            nn.Sigmoid() # 0-1の範囲に出力
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x).view(b, t, -1)

        # Transformerで時系列の特徴を統合
        z = self.transformer(features) # [b, t, 512]

        # 各フレームを復元
        reconstructed = self.decoder(z) # [b, t, 224*224]
        return reconstructed.view(b, t, 1, 224, 224) # モノクロ画像として復元
