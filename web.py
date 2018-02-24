# -*- coding: utf-8 -*-

# References:
#   A fileupload app with flask
#   https://github.com/arvelt/flask-fileupload-sample
#   Flaskで画像アップローダー
#   https://qiita.com/Gen6/items/f1636be0fe479f42b3ee

import os
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug import secure_filename
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
      filename = secure_filename(file.filename)
      img = Image.open(file)
      infer = inferFromImage(img)
      result = {
        "Result": {
          "filename": filename,
          "infer": infer
        }
      }
      return jsonify(ResultSet=result)

    else:
      return ''' <p>許可されていない拡張子です</p> '''
  else:
    return redirect(url_for('index'))

if __name__ == '__main__':
  app.run(debug=True)
