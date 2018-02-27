# -*- coding: utf-8 -*-

# References:
#   A fileupload app with flask
#   https://github.com/arvelt/flask-fileupload-sample
#   Flaskで画像アップローダー
#   https://qiita.com/Gen6/items/f1636be0fe479f42b3ee

import os
import base64
from io import BytesIO
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image

from mnist import inferFromImage

ALLOWED_EXTENSIONS = set(['PNG', 'png', 'JPG', 'jpg', 'jpeg'])

app = Flask(__name__)

def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
  if request.method == 'POST':
    file = request.files['img_file']
    if file and allowed_file(file.filename):
      img = Image.open(file)
      infer, outputs, inputImg = inferFromImage(img)

      # (出力値, ラベル)のリストに変換して出力値の降順に整列
      outputs = [(outputs[i], i) for i in range(len(outputs))]
      outputs = sorted(outputs, reverse=True)

      # 画像をdata URIに変換
      buf = BytesIO()
      inputImg.save(buf, format='PNG')
      imgStr = base64.b64encode(buf.getvalue())
      imgStr = 'data:image/png;base64,' + imgStr.decode('utf-8')

      result = {
        "Result": {
          "filename": file.filename,
          "infer": infer,
          "values": outputs,
          "image": imgStr
        }
      }
      return jsonify(ResultSet=result)

    else:
      return ''' <p>許可されていない拡張子です</p> '''
  else:
    return redirect(url_for('index'))

if __name__ == '__main__':
  app.run(debug=True)
