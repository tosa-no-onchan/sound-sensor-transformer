import numpy as np
import cv2
import librosa
#from moviepy.editor import VideoFileClip
from moviepy import VideoFileClip,AudioFileClip
import os

def video_to_spectrogram_sequence(video_path, n_seconds=4, L_frames=8, hop_length=49):
    sr = 22050
    target_len = n_seconds * sr
    # 1. 音声の読み込み (拡張子で自動切り替え)
    try:
        ext = os.path.splitext(video_path)[1].lower()
        # 拡張子が wav なら AudioFileClip、それ以外（mp4等）なら VideoFileClip
        if ext == ".wav":
            clip = AudioFileClip(video_path)
        else:
            clip = VideoFileClip(video_path)
        with clip:
            duration = min(clip.duration, n_seconds)
            # MoviePy v2.0以降の書き方
            audio_clip = clip.audio.subclipped(0, duration) if hasattr(clip, 'audio') and clip.audio else clip.subclipped(0, duration)
            # 指定サンプリングレートでNumPy配列化
            y = audio_clip.to_soundarray(fps=sr)
            # ステレオ(2ch)なら平均をとってモノラル(1ch)に
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            # 長さ調整（パディングまたはカット）
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            clip.close()
    except Exception as e:
        print(f"Audio extraction error: {e}")
        y = np.zeros(target_len)

    # add by nishi 2026.4.17
    # 音声を -1.0 〜 1.0 の範囲に正規化する
    if True:
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

    # 2. メルスペクトログラムの計算
    # 黒パディングを消すために hop_length=49 を使用
    #hop_length = 49
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_mels=224, hop_length=hop_length)

    S_dB = librosa.power_to_db(S, ref=np.max)
    # 3. データの正規化 (-80dB〜0dB を 0〜255に)
    S_norm = ((S_dB + 80) / 80 * 255).clip(0, 255).astype(np.uint8)

    # 16枚で224幅を切り出すための計算 (1792 / 16 = 112)
    #if L_frames==8:
    #    step = 224
    #if L_frames==16:
    #    step = 112
    step=int(1792/ L_frames)

    # 4. 時間軸方向にL_frames個の画像に分割
    # hop_length=49 なら total_width はほぼ 1792 (224*8) になります
    spectrogram_frames = []
    for i in range(L_frames):
        start = i * step
        end = start + 224
        frame = S_norm[:, start:end]

        # 端っこで幅が足りない場合はパディング
        if frame.shape[1] < 224:
            frame = np.pad(frame, ((0,0), (0, 224 - frame.shape[1])), mode='constant')

        # 1. RGB化 [H, W, C]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # 2. 【重要】正規化 (0-255 -> 0-1 & mean/std)
        # 後の可視化でエラーが出ないよう、ここで計算を済ませるのが安全です
        frame_float = frame_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_float - mean) / std

        # 3. 【ここを修正】[H, W, C] -> [C, H, W] に変換
        frame_final = frame_normalized.transpose(2, 0, 1)
        spectrogram_frames.append(frame_final)

    return np.array(spectrogram_frames,dtype=np.float32) # 戻り値は [L, 3, 224, 224]
