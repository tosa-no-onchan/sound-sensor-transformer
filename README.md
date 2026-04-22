# sound-sensor-transformer  

機械の音で、故障の診断 Sound Sensor Transformer.  
  
CNN-Transformer ハイブリッドモデル で、動画のクラス分類をする。  
かって、CNNとLSTMを組み合わせたモデルの「LRCN (Long-term Recurrent Convolutional Networks)」の、 LSTM 部分を、  
Transformer に置き換えたモデル。第2段。  
今回は、上記をベースに、  
教師なし Transformer (Temporal AutoEncoder) を使います。  

#### 1. Train  
  datasets/bike/normal に、バイクのアイドリングの動画(mp4) 4秒以上 を集めて、学習させます。  
  教師なし学習なので、アイドリングの動画だけ集めます。  
  
sound_sensor_transformer_train.ipynb  

#### 2. Test  
  datasets/bike/speed に、バイクの加速や、アクセルを空ぶかしする動画を集めます。 

#### 3. ONNX 変換  
  $ python sound_sensor_torch2onnx_for_pc.py  

#### 4. ONNX でのテスト  
  $ sound_sensor_orangepi_onnx.py  
  
