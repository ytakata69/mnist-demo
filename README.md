# mnist-demo

ニューラルネット・ライブラリ [Chainer](https://chainer.org) と手書き数字データ集合 [MNIST](http://yann.lecun.com/exdb/mnist/) を使ったお遊びデモ。

描画枠に数字を手書きして送信すると，判別結果（0〜9のいずれか）が表示される。

[デモサイト](https://dry-sierra-33432.herokuapp.com) on [Heroku](https://heroku.com)


## 学習済みニューラルネットによる推論

* `mnist.py` - 推論プログラム
* `data/mnist.model` - 学習済みパラメータ

MNISTを使って訓練したニューラルネットを使って，入力画像から推論結果 (0〜9) を計算する（`mnist.py` を単独で実行すると，`data/three.png` に対する推論結果を出力する）。

ほぼ[チュートリアル](https://qiita.com/mitmul/items/1e35fba085eb07a92560)のプログラム例の通り（PNG画像をNumPy配列に変換する部分は後述）。

### 学習

学習は，[chainer.git](https://github.com/chainer/chainer.git) の `examples/mnist/train_mnist.py` を，無改造・オプション指定なしで実行しただけ（GPUがなくても6〜7分で終わる）。

* 中間層2層, それぞれ1000ノード, 活性化関数はReLU
* 更新アルゴリズムはAdam
  （勾配降下法の改良の一つ．
  [参考記事](https://postd.cc/optimizing-gradient-descent/)）
* 損失関数はsoftmax cross entropy
* 20エポック
* (テストデータの正解率 98.2%)

`train_mnist.py` による学習結果は `result/snapshot_iter_12000` というファイルに保存される。このファイル (snapshotファイル) はネットワークパラメータ以外の情報も含んでいるので，以下のようなコードを実行してネットワークパラメータだけ取り出す（`train_mnist_custom_loop.py` を使った場合はsnapshotファイルではなくmodelファイルが出力される）。

```python
from train_mnist import MLP
from chainer import serializers

n_units = 1000
net = MLP(n_units, 10)

serializers.load_npz(
  'result/snapshot_iter_12000',
  net,
  path='updater/model:main/predictor/')

serializers.save_npz('mnist.model', net)
```

### 画像の整形とNumPy配列への変換

MNISTに合わせて，入力画像を 28&times;28 = 784 画素 (0.0〜1.0, 背景が0.0) の配列に変換しなければならない。
画像の変換に [Pillow](https://pillow.readthedocs.io/en/latest/) を使っている。

[MNIST](http://yann.lecun.com/exdb/mnist/) 配布元の説明に従って，以下の前処理を実行（`mnist.py`の`inferFromImage`を参照）。重心の計算はNumPyで行っている。

1. 余白を除く
2. 20&times;20の矩形にぴったり合うようアスペクト比を変えずに大きさを調節
3. 28&times;28の矩形の中に重心 (center of mass) を中心にして配置


## Webアプリ化

* `web.py` - Webアプリのメインプログラム
* `templates/index.html` - メインページのHTML
* `static/*` - JavaScriptやCSSなどの静的ファイル

PythonベースWebフレームワーク
[Flask](http://flask.pocoo.org)
を使ってWebアプリ化し，
[Heroku](https://heroku.com) に載せる。

`web.py` の機能は以下の2つだけ。数十行でできている。

* `/` がアクセスされるとメインページを返す。
* `/send` にてPNGまたはJPEGファイルのアップロードを受け付け，`mnist.py` の `inferFromImage` を呼び出して，実行結果をJSONで返す。

### drawingboard.js による手書きUI

`index.html` に
[drawingboard.js](https://github.com/Leimi/drawingboard.js) による手書きユーザインタフェースを取り付けた。

[drawingboard.jsのデモページ](http://leimi.github.io/drawingboard.js/)

drawingboard.js を使うのに [jQuery](https://jquery.com) が必要。

drawingboard.js で描いた画像は [data URI](https://ja.wikipedia.org/wiki/Data_URI_scheme) 形式で取り出されるが，これをファイルアップロード形式で送信するのが若干面倒だった（[参考記事](https://stackoverflow.com/questions/4998908)）。`web.py` を変更して data URI を受け取るようにする，という選択肢もある。

### Heroku へのデプロイ

* `Procfile` - プロセス定義ファイル
  ```
  web: gunicorn web:app --log-file=-
  ```
* `runtime.txt` - 実行環境指定ファイル (Pythonのバージョンを指定)
  ```
  python-3.6.4
  ```
* `requirements.txt` - 依存ライブラリのリスト
  ```
  chainer==3.2.0
  Flask==0.12.2
  gunicorn==19.7.1
  numpy==1.14.1
  Pillow==5.0.0
  ```

上記のファイルを置いておけば，Heroku が勝手に必要なライブラリをインストールしてWebサーバ ([Gunicorn](http://gunicorn.org)) を起動してサービスを開始する。

デプロイの前に手元でテスト実行する場合は，下記を実行すればよい。`http://127.0.0.1:5000/` でアクセスできるサーバが起動する。

```bash
$ pip install -r requirements.txt  # 依存ライブラリをインストール
$ python3 web.py                   # サーバを起動
```

## 使用ライブラリ

* [Chainer](https://chainer.org) - ニューラルネットワーク・ライブラリ
* [Pillow](https://pillow.readthedocs.io/en/latest/) - PIL (Python Imaging Library) 互換ライブラリ
* [Flask](http://flask.pocoo.org) - Pythonベースの軽量Webフレームワーク
* [jQuery](https://jquery.com) - JavaScript拡張ライブラリ．drawingboard.jsが依存
* [drawingboard.js](https://github.com/Leimi/drawingboard.js#drawingboardjs) - HTML5 canvasベースのドローソフト

## 参考資料

* [Chainer v3 ビギナー向けチュートリアル](https://qiita.com/mitmul/items/1e35fba085eb07a92560)
* 実践! GPUサーバでディープラーニング, 長谷川猛, [Software Design 2018年3月号](http://gihyo.jp/magazine/SD/archive/2018/201803)
* [Tensorflow, MNIST and your own handwritten digits](http://opensourc.es/blog/tensorflow-mnist)
