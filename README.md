# mnist-demo

ニューラルネット・ライブラリ [Chainer](https://chainer.org) と手書き数字データ集合 [MNIST](http://yann.lecun.com/exdb/mnist/) を使ったお遊びデモ。

描画枠に数字を手書きして送信すると，判別結果（0〜9のいずれか）が表示される。

[デモサイト](https://dry-sierra-33432.herokuapp.com) on [Heroku](https://heroku.com)


## 学習済みニューラルネットによる推論

* `mnist.py` - 推論プログラム
* `data/mnist.model` - 学習済みパラメータ

MNISTを使って訓練したニューラルネットを使って，入力画像から推論結果 (0〜9) を計算する（`mnist.py` を単独で実行すると，`data/three.png` に対する推論結果を出力する）。

ほぼ[チュートリアル](https://qiita.com/mitmul/items/1e35fba085eb07a92560)通りの内容（PNG画像をNumPy配列に変換する部分は後述）。

### 学習

学習は，[chainer.git](https://github.com/chainer/chainer.git) の `examples/mnist/train_mnist.py` を実行するだけ（GPUがなくても6〜7分で終わる）。

`train_mnist.py` による学習結果は `result/snapshot_iter_12000` というファイルに保存される。このファイル (snapshotファイル) はネットワークパラメータ以外の情報も含んでいるので，以下のようなコードを実行してモデルパラメータだけ取り出す（`train_mnist_custom_loop.py` を使った場合はsnapshotファイルではなくmodelファイルが出力される）。

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

（「描画内容が中心に大きく」収録されるよう補正した方が判別精度が上がると思うが，さぼっている。）

## Webアプリ化

* `web.py` - Webアプリのメインプログラム
* `templates/index.html` - メインページのHTML
* `static/*` - JavaScriptやCSSなどの静的ファイル

PythonベースWebフレームワーク
[Flask](http://flask.pocoo.org)
を使ってWebアプリ化し，
[Heroku](https://heroku.com) に載せる。

`web.py` の機能は以下の2つだけ。数十行でできている。

* URL `/` がアクセスされるとメインページを返す。
* URL `/send` にてPNGまたはJPEGファイルのアップロードを受け付け，`mnist.py` の `inferFromImage` を呼び出して，実行結果をJSONで返す。

### drawingboard.js による手書きUI

`index.html` に
[drawingboard.js](https://github.com/Leimi/drawingboard.js) による手書きユーザインタフェースを取り付けた。

[drawingboard.jsのデモページ](http://leimi.github.io/drawingboard.js/)

drawingboard.js を使うのに [jQuery](https://jquery.com) が必要。
ついでなので，`web.py` への画像判別リクエストに [jQuery.ajax()](http://api.jquery.com/jquery.ajax/) を使っている。

drawingboard.js で描いた画像は [data URI](https://ja.wikipedia.org/wiki/Data_URI_scheme) 形式で取り出されるが，これをファイルアップロード形式で送信するのがちょっと面倒だった。`web.py` を変更して data URI を受け取るようにする，という選択肢もある。

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


## 参考資料

* [Chainer v3 ビギナー向けチュートリアル](https://qiita.com/mitmul/items/1e35fba085eb07a92560)
* 実践! GPUサーバでディープラーニング, 長谷川猛, [Software Design 2018年3月号](http://gihyo.jp/magazine/SD/archive/2018/201803)
