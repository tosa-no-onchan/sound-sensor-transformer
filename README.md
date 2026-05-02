# sound-sensor-transformer  

機械の音で、故障の診断 Sound Sensor Transformer.  
  
CNN-Transformer ハイブリッドモデル で、動画のクラス分類をする。  
かって、CNNとLSTMを組み合わせたモデルの「LRCN (Long-term Recurrent Convolutional Networks)」の、 LSTM 部分を、  
Transformer に置き換えたモデル。第2段。  
今回は、上記をベースに、  
教師なし Transformer (Temporal AutoEncoder) を使います。  

#### 1. Train  
  datasets/bike/normal に、バイクのアイドリングの動画(mp4) 4秒以上 を集めて、学習させます。  
  始めの、4[秒] 部分だけ、使います。  
  教師なし学習なので、アイドリングの動画だけ集めます。  
  
sound_sensor_transformer_train.ipynb  

#### 2. 検証  
  sound_sensor_transformer_predict_ex.ipynb

#### 3. Test用 データ 
  datasets/bike/speed に、バイクの加速や、アクセルを空ぶかしする動画を集めます。 

#### 4. ONNX 変換  
  $ python sound_sensor_torch2onnx_for_pc.py  

#### 5. ONNX でのテスト  
  $ sound_sensor_orangepi_onnx.py  
  
#### 6. 参照  
[機械の音で、故障の診断 Sound Sensor Transformer.](https://www.netosa.com/blog/2026/04/-sound-sensor-transformer.html)  

#### 7. update  
  2026.5.2  
  model の outputs の変更をしました。  
  1) 1 channel -&gt; 3 channels に、拡張する。
  2) sigmoid() を通さない、または、orch.nn.functional.softplus() を通す。
     
  decoder の修正。  
  sigmoid は、使わない。 nn.Linear() を、3channels 対応にする。  
````
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.channels * self.n_mels * self.n_mels),
            #nn.Sigmoid() # 0-1の範囲に出力
        )

````

  forward の修正。  
  sigmoid を、使わないか、torch.nn.functional.softplus() を使う。  
````
    def forward(self, x):
        ....        
        if not self.use_sigmoid:
            reconstructed = self.decoder(z)
        else:
            reconstructed_raw = self.decoder(z)
            # 注) sgimoid を、入れると、loss=0.199 で、止まる。
            # バイクのシャリシャリを拾うなら、sigmoid は、使わない。 by nishi 2026.5.1
            #reconstructed = torch.sigmoid(reconstructed_raw)
            # 0〜1に押し込めるのではなく、単にマイナスを消して「やりすぎ」を許容する
            reconstructed = torch.nn.functional.softplus(reconstructed_raw) 
        return reconstructed.view(b, t, self.channels, self.n_mels, self.n_mels) # color or モノクロ画像として復元

````
