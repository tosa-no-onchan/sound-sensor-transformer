# sound_sensor_orangepi_onnx.py
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import os
import sys

import cv2
import numpy as np

from video_to_spectrogram_sequence import video_to_spectrogram_sequence

def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    # リサイズ
    resized = cv2.resize(image, (new_w, new_h))
    # 黒埋め用のキャンバス作成
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    # 中央に配置
    offset_y = (target_size[0] - new_h) // 2
    offset_x = (target_size[1] - new_w) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    return canvas


def preprocess_images_numpy(frames):
    """ResNet18 (ImageNet) の前処理をNumPyで完全再現"""
    # 1. 0-255 (int) -> 0-1 (float32)
    images = np.array(frames).astype(np.float32) / 255.0
    
    # 2. ImageNetの正規化パラメータ
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 3. 正規化計算: (画像 - 平均) / 標準偏差
    # NumPyのブロードキャスト機能を利用
    images = (images - mean) / std
    
    # 4. 軸入れ替え: (Time, H, W, C) -> (Time, C, H, W)
    images = images.transpose(0, 3, 1, 2)
    
    # 5. バッチ次元追加: (1, Time, C, H, W)
    return np.expand_dims(images, axis=0)


class SoundSensorONNX:
    def __init__(self, model_path):
        # CPUの4コアをフルに使う設定
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path,
                            sess_options=options, 
                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, video_path, n_seconds=4, L_frames=8, hop_length=49):
        # 前回作成した MoviePy + librosa の処理
        # (ここでは中身を省略しますが、[1, 8, 3, 224, 224] の float32 NumPyを返す)
        # ※ hop_length=49 を忘れずに！
        in_data = video_to_spectrogram_sequence(video_path, n_seconds=n_seconds, L_frames=L_frames,hop_length=hop_length)
        input_data = np.expand_dims(in_data, axis=0)
        return input_data
        #pass 

    def run_inference(self, input_data):
        # input_data: [1, 16, 3, 224, 224]
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_data})
        end = time.perf_counter()
        reconstructed = outputs[0]

        if False:
            # 各フレームごとのMSEを計算: [1, 16, 1, 224, 224] -> 各フレーム(16枚)の平均誤差
            # MSE計算 (targetはinputのRチャンネル)
            target = input_data[:, :, 0:1, :, :]

            # axis=(2,3,4) で各フレーム(C,H,W)ごとの誤差を出す
            frame_losses = np.mean((reconstructed - target) ** 2, axis=(2, 3, 4))
        else:
            # target を作るとき、明示的にスライスした後に copy() してメモリ配置を整える
            target = input_data[:, :, 0:1, :, :].copy() 
            # 差分を計算
            diff_sq = (reconstructed - target) ** 2

            # フレームごとのLoss [Batch, Time]
            # axis=(2, 3, 4) は [Channel, Height, Width] の軸をすべて平均化するという意味です
            frame_losses = np.mean(diff_sq, axis=(2, 3, 4))[0] 

        #loss = np.mean((reconstructed - target) ** 2)

        # 【ここが重要】平均ではなく最大値を取る
        max_loss = np.max(frame_losses)
        avg_loss = np.mean(frame_losses) # 参考に平均も出す

        #return loss, end - start
        return max_loss, avg_loss, end - start


        if False:
            #----
            # target を作るとき、明示的にスライスした後に copy() してメモリ配置を整える
            target = input_data[:, :, 0:1, :, :].copy() 

            # 差分を計算
            diff_sq = (reconstructed - target) ** 2

            # フレームごとのLoss [Batch, Time]
            # axis=(2, 3, 4) は [Channel, Height, Width] の軸をすべて平均化するという意味です
            frame_losses = np.mean(diff_sq, axis=(2, 3, 4))[0] 

            max_loss = np.max(frame_losses)
            avg_loss = np.mean(frame_losses)

    def __call__(self,video_path, n_seconds=4, L_frames=8, hop_length=49):
        in_data = video_to_spectrogram_sequence(video_path, n_seconds=n_seconds, L_frames=L_frames,hop_length=hop_length)
        input_data = np.expand_dims(in_data, axis=0)
        return self.run_inference(input_data)


class ONNXPredictor:
    def __init__(self, onnx_model_path, class_names, n_seconds=4, L_frames=8):
        # 1. ONNX Runtime セッションの作成
        # Orange Pi 5 の場合は 'CPUExecutionProvider' が基本ですが、
        # PCなら 'CUDAExecutionProvider' (GPU) も使えます。
        # ONNXセッションの初期化 (CPU専用)
        # Orange PiのCPUリソースをフル活用するため、セッションオプションを設定
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # Orange Piのコア数に合わせて調整
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=options, 
            providers=['CPUExecutionProvider'] 
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.n_seconds = n_seconds
        self.L_frames = L_frames

        if False:
          self.padder = AspectRatioPad(size=(224, 224)) # 前回のカスタムクラス
          # 前処理 (Normalize)
          self.normalize = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])

    def predict(self, video_path):
        if False:
          # 2. 動画の読み込みとサンプリング
          video, _, _ = read_video(video_path, start_pts=0, end_pts=self.n_seconds, pts_unit='sec')
          total_frames = video.shape[0]

          indices = torch.linspace(0, total_frames - 1, steps=self.L_frames).long()
          sampled_video = video[indices]

          # 3. 前処理 (numpy配列として整形)
          processed_frames = []
          for frame in sampled_video:
              img = Image.fromarray(frame.numpy())
              img = self.padder(img)
              #img2 = resize_with_padding(img)
              img = self.normalize(img) # Tensor化 + 正規化
              processed_frames.append(img.numpy())

          # [1, Time, C, H, W] の形にする
          input_data = np.stack(processed_frames)[np.newaxis, ...]

        else:
          # --- 1. 映像読み込み (OpenCV) ---
          cap = cv2.VideoCapture(video_path)
          # (中略: indices計算と8フレーム抽出。以前のresize_with_paddingを使用)
          fps = cap.get(cv2.CAP_PROP_FPS) # 1秒あたりのフレーム数
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

          # --- 映像も「最初の3秒」に限定する ---
          #max_duration = 3.0
          # 3秒分、または動画全体の短い方のフレーム数をターゲットにする
          end_frame = min(total_frames, int(self.n_seconds * fps))

          # 0フレームから3秒地点（end_frame）の間で16枚抜く
          indices = np.linspace(0, end_frame - 1, self.L_frames).astype(int)

          frames = []
          for i in indices:
              cap.set(cv2.CAP_PROP_POS_FRAMES, i)
              ret, frame = cap.read()
              if ret:
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  # 1. アスペクト比維持リサイズ
                  frame = resize_with_padding(frame)
                  frames.append(frame)
              else:
                  frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
          cap.release()
          # --- 2. 前処理 (NumPyのみ) ---
          input_data = preprocess_images_numpy(frames)
          #print('pixel_values.dtype:',pixel_values.dtype)

        # 4. ONNX 推論
        start_time = time.perf_counter()
        
        # 入力名をキーにした辞書でデータを渡す
        outputs = self.session.run(None, {self.input_name: input_data})
        
        inference_time = time.perf_counter() - start_time
        
        # 5. 結果の解析
        logits = outputs[0]
        pred_idx = np.argmax(logits, axis=1)[0]
        confidence = self._softmax(logits[0])[pred_idx]

        return self.class_names[pred_idx], confidence, inference_time

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

if __name__ == '__main__':
    MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/sound-sensor-transformer/sound_sensor_model_fixed-8_4.onnx"
    #MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/sound-sensor-transformer/sound_sensor_model_fixed-16_4.onnx"
    # --- 実行と計測 ---
    tester = SoundSensorONNX(MODEL_PATH)
    if False:
        # テスト用データ（1回目は初期化が含まれるため、複数回計測がおすすめ）
        dummy_data = np.random.randn(1, 8, 3, 224, 224).astype(np.float32)

        print("速度計測開始...")
        latencies = []
        for i in range(10):
            score, t = tester.run_inference(dummy_data)
            latencies.append(t)
            print(f"回数 {i+1}: 推論時間 {t:.4f}秒 | スコア {score:.6f}")

        print(f"\n平均推論時間: {np.mean(latencies):.4f}秒")
        sys.exit()

    if False:
        anomaly_video_path = "datasets/bike/speed/yL7jbivvApg_trim3.mp4"
        print("速度計測開始...")
        latencies = []
        for i in range(10):
            # (1, 8, 3, 224, 224)
            #input_data=tester.preprocess(anomaly_video_path)
            #score, t = tester.run_inference(input_data)
            score, t = tester(anomaly_video_path)
            latencies.append(t)
            print(f"回数 {i+1}: 推論時間 {t:.4f}秒 | スコア {score:.6f}")

        print(f"\n平均推論時間: {np.mean(latencies):.4f}秒")
        sys.exit()


    # --- 1. ファイルパスとラベルのリストを作成 ---
    data_dir = "datasets/bike"
    flist=os.listdir(data_dir)
    cnt=0

    import time

    # In[ ]:
    CLASS_NAMES=['normal','speed']
    #video_path=data_dir+'/'+flist[cnt]
    if True:
        latencies = []
        for sub_dir in CLASS_NAMES:
            data_dir = os.path.join("datasets/bike", sub_dir)
            #flist=os.listdir(data_dir)
            flist = [f for f in os.listdir(data_dir) if f.endswith(".mp4") or f.endswith(".wav")]
            p_num = min(len(flist),100)
            print('-----')
            cnt=0
            for i in range(p_num):
                video_path=data_dir+'/'+flist[i]
                #print("flist[i]:",flist[i])
                if not flist[i] == "backup":
                    print("video_path:",video_path)
                    cnt+=1
                    #result, confidence=predict_video_fast(video_path, num_frames=num_frames, max_duration=max_duration)
                    #label, conf, t = predictor.predict(video_path)
                    max_loss, avg_loss,t = tester(video_path,L_frames=8)
                    #score, t = tester(video_path,L_frames=16)
                    latencies.append(t)
                    #print(f"回数 {cnt}: 推論時間 {t:.4f}秒 | スコア {score:.6f}")
                    print(f"回数 {cnt}: 推論時間 {t:.4f}秒 | max_loss {max_loss:.6f} | avg_loss {avg_loss:.6f}")


                    #print('result:',result, 'confidence:',confidence)
                    #print(f"結果: {label} ({conf:.2%})")
                    #print(f"ONNX推論時間: {t:.4f} 秒")

        print(f"\n平均推論時間: {np.mean(latencies):.4f}秒")


    #label, conf, t = predictor.predict("test_video.mp4")

    #print(f"結果: {label} ({conf:.2%})")
    #print(f"ONNX推論時間: {t:.4f} 秒")

