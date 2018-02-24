# -*- coding: utf-8 -*-

# 学習済みモデルを使って推論する
#
# Chainer v3 ビギナー向けチュートリアル
# https://qiita.com/mitmul/items/1e35fba085eb07a92560

# Chainerで画像を読み込む際のTips
# https://qiita.com/ysasaki6023/items/fa2fe9c2336677821583
# PIL/Pillow チートシート
# https://qiita.com/pashango2/items/145d858eff3c505c100a

# ネットワークを表すコード

import chainer
import chainer.links as L
import chainer.functions as F

# Link = パラメータを持つ関数
# Function = パラメータを持たない関数
# Chain = パラメータを持つ層 (Link) をまとめておくためのクラス

# 多層パーセプトロン (Multilayer Perceptron)
class MLP(chainer.Chain):

  def __init__(self, n_mid_units=1000, n_out=10):
    super(MLP, self).__init__()

    # パラメータを持つ層の登録
    with self.init_scope():
      self.l1 = L.Linear(None, n_mid_units)
      self.l2 = L.Linear(n_mid_units, n_mid_units)
      self.l3 = L.Linear(n_mid_units, n_out)

  def __call__(self, x):
    # データを受け取った際のforward計算を書く
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    return self.l3(h2)

def infer(x, gpu_id=-1):
  from chainer.cuda import to_cpu
  from chainer import serializers

  infer_net = MLP()
  serializers.load_npz('data/mnist.model', infer_net)

  if gpu_id >= 0:	# CPUで計算したい場合は-1
    infer_net.to_gpu(gpu_id)

  x = x[None, ...]	# ミニバッチの形にする

  # ネットワークと同じデバイス上にデータを送る
  x = infer_net.xp.asarray(x)

  y = infer_net(x)	# モデルのforward関数に渡す
  y = y.array		# Variable形式で出てくるので中身を取り出す
  y = to_cpu(y)		# 結果をCPUに送る

  return y.argmax(axis=1)[0]

def inferFromImage(img):
  from PIL import Image, ImageOps
  import numpy as np

  img = img.convert('L')	# grayscale
  img = img.resize((28, 28), Image.ANTIALIAS)
  img = ImageOps.invert(img)	# negate
  #img.show()

  # NumPy配列に変換
  arrayImg = np.asarray(img).astype(np.float32) / 255.

  return int(infer(arrayImg))

if __name__ == '__main__':
  from PIL import Image

  img = Image.open('data/three.png')
  print('予測ラベル:', inferFromImage(img))
