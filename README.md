# Informative-Drawings-ONNX-Sample
<img src="https://user-images.githubusercontent.com/37477845/196734419-c14de248-5440-4bca-b19d-d3c2d94fe391.gif" width="85%"><br>
[Informative Drawings](https://github.com/carolineec/informative-drawings)のPythonでのONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>
変換自体を試したい方はColaboratoryなどで[Informative-Drawings-Convert2ONNX.ipynb](Informative-Drawings-Convert2ONNX.ipynb)を使用ください。<br>

# Requirement(ONNX推論)
* OpenCV 4.5.3.56 or later
* onnxruntime-gpu 1.9.0 or later <br>※onnxruntimeでも動作しますが、推論時間がかかるのでGPUをお勧めします

# Demo
デモの実行方法は以下です。
```bash
python sample_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/opensketch_style_512x512.onnx
* --input_shape<br>
モデルの入力サイズ<br>
デフォルト：512

# Reference
* [informative-drawings](https://github.com/carolineec/informative-drawings)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Informative-Drawings-ONNX-Sample is under [MIT License](LICENSE).

# License(Image)
女性の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
