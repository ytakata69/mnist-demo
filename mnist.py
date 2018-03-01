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

from chainer.cuda import to_cpu
from chainer import serializers

infer_net = MLP()
serializers.load_npz('data/mnist.model', infer_net)

gpu_id = -1
if gpu_id >= 0:	# CPUで計算したい場合は-1
  infer_net.to_gpu(gpu_id)


def infer(x):
  """
  入力xに対する推論結果を返す
  :param numpy.ndarray x: 入力
  :rtype:  (int, [float])
  :return: (予測ラベル, [各ラベルに対する出力値])
  """
  x = x[None, ...]	# ミニバッチの形にする

  # ネットワークと同じデバイス上にデータを送る
  x = infer_net.xp.asarray(x)

  y = infer_net(x)	# モデルのforward関数に渡す
  y = y.array		# Variable形式で出てくるので中身を取り出す
  y = to_cpu(y)		# 結果をCPUに送る

  return (int(y.argmax(axis=1)[0]), y.tolist()[0])

def inferFromImage(img):
  """
  画像imgに対する推論結果を返す
  :param PIL.Image.Image img: 画像
  :rtype:  (int, [float], PIL.Image.Image)
  :return: (予測ラベル, [各ラベルに対する出力値], 正規化画像)
  """
  from PIL import Image, ImageOps
  import numpy as np

  img = img.convert('L')	# grayscale
  img = ImageOps.invert(img)	# negate

  # 前処理: 非0領域をアスペクト比を変えずに20x20の矩形に収め，
  # 重心を中心にして28x28にする
  # References:
  #   http://opensourc.es/blog/tensorflow-mnist

  img = img.crop(box=img.getbbox())	# crop
  wd, ht = img.size

  # アスペクト比を変えずに20x20の矩形に収める
  if wd < ht:
    wd = wd * 20 // ht
    ht = 20
  else:
    ht = ht * 20 // wd
    wd = 20
  img = img.resize((wd, ht), Image.ANTIALIAS)

  cx, cy = centerOfMass(img)	# 重心

  # 重心を中心にして28x28の矩形の中に配置
  bgImg = Image.new(img.mode, (28, 28), color=0)
  ox = -cx + 28 // 2
  oy = -cy + 28 // 2
  bgImg.paste(img, (ox, oy))
  img = bgImg

  # NumPy配列に変換
  arrayImg = np.asarray(img).astype(np.float32) / 255.

  return infer(arrayImg) + (img,)

def centerOfMass(img):
  """
  画像imgの重心位置を返す
  :param PIL.Image.Image img: グレイスケール画像
  :rtype:  (int, int)
  :return: (重心のX座標, 重心のY座標)
  """
  import numpy as np

  m = np.asarray(img)	# NumPy配列に変換
  ht, wd = m.shape
  sum = np.sum(m)

  # https://stackoverflow.com/questions/37519238
  dx = np.sum(m, axis=0)	# 各列の合計からなるベクトル
  dy = np.sum(m, axis=1)	# 各行の合計からなるベクトル

  # np.arange(wd) == [0, 1, 2, ..., wd-1]
  cx = np.sum(dx * np.arange(wd)) / sum
  cy = np.sum(dy * np.arange(ht)) / sum

  return (int(np.rint(cx)), int(np.rint(cy)))

if __name__ == '__main__':
  from PIL import Image

  img = Image.open('data/three.png')
  label, outputs, img = inferFromImage(img)
  img.show()
  print('予測ラベル:', label)
