import torch
import torch.nn as nn

import numpy as np
import sys

from VideoAutoEncoder import VideoAutoEncoder

# インポートをシンプルにします
#from torch.nn.attention import sdpa_kernel, SDPBackend


def export_to_onnx_for_rknn(model, save_path="sound_sensor_transformer.onnx"):
    model.eval()
    model.to("cpu")
    # 2. ダミーデータの作成
    # [Batch:1, Time:8, Channel:3, Height:224, Width:224]
    dummy_input = torch.randn(1, 8, 3, 224, 224)
    #dummy_input = torch.randn(1, 16, 3, 224, 224)


    if True:
        torch.onnx.export(
            model,
            dummy_input,                # モデルへの入力（サンプル）
            save_path,
            export_params=True,        # 重みをファイルに書き込む
            opset_version=14,          # DETRの演算をサポートするバージョン
            #opset_version=17,          # DETRの演算をサポートするバージョン
            do_constant_folding=True,  # 定数畳み込みでグラフを最適化
            input_names=['input'],
            output_names=['output']
            # dynamic_axes はあえて指定せず、サイズを 480x480 に固定します（RKNN向け）
        )

    print(f"✅ RKNN変換用ONNX を保存しました: {save_path}")

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#MODEL_PATH = "output-16frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-8frame3sec/best_loss_multimodal_model.pth"  # 保存したモデルのパス
MODEL_PATH = "output-8frame4sec/best_loss_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-8frame4sec/latest_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-16frame4sec/latest_model.pth"  # 保存したモデルのパス
#MODEL_PATH = "output-8frame4sec-full-scratch/best_loss_multimodal_model.pth"

save_path = "sound_sensor_model_fixed-8_4.onnx"
#save_path = "sound_sensor_model_fixed-16_4.onnx"

# --- 使い方 ---
# model = YourModelClass(...) # 以前定義したモデルクラスをインスタンス化
model = VideoAutoEncoder()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

export_to_onnx_for_rknn(model, save_path=save_path)


sys.exit()

