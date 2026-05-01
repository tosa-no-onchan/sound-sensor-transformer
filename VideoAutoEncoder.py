
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F

USE_SIGMOID=True

class VideoAutoEncoder(nn.Module):
    def __init__(self, num_frames=8, n_mels=224, use_full_scratch=False,channels=3): # 16に変更  --> num_frames は、使っていない!!
        super(VideoAutoEncoder, self).__init__()
        # エンコーダ（ResNet + Transformer）
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if not use_full_scratch:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.n_mels=n_mels
        self.d_model=512
        self.channels=channels      # gray:1 / color:3

        # Transformerの定義
        #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, 
                    nhead=8, 
                    batch_first=True),
                    num_layers=3)

        # デコーダ（512次元のベクトルから 224x224 の画像を復元する簡易的な構成）
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.channels * self.n_mels * self.n_mels),
            #nn.Sigmoid() # 0-1の範囲に出力
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x).view(b, t, -1)

        # Transformerで時系列の特徴を統合
        z = self.transformer(features) # [b, t, 512]

        if not USE_SIGMOID:
            # 各フレームを復元 [b, t, n_mels*n_mels]
            reconstructed = self.decoder(z)
        else:
            # デコーダーを通す（この時点では生の値 = Logits）
            reconstructed_raw = self.decoder(z)
            # 最後に一括で Sigmoid を適用（メリハリを保つ）
            # 注) sgimoid を、入れると、loss=0.199 で、止まる。
            # バイクのシャリシャリを拾うなら、sigmoid は、使わない。 by nishi 2026.5.1
            #reconstructed = torch.sigmoid(reconstructed_raw)
            # 0〜1に押し込めるのではなく、単にマイナスを消して「やりすぎ」を許容する
            reconstructed = torch.nn.functional.softplus(reconstructed_raw) 
        return reconstructed.view(b, t, self.channels, self.n_mels, self.n_mels) # color or モノクロ画像として復元
