from flask import Flask, request, jsonify, send_file
import os
import urllib2
import base64
import numpy as np
import mxnet as mx
# import Image

import nstyle_forward

MODEL_DIR = 'pretrained'

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def hello_world():
  if request.method == 'POST':
    return 'hello world'

@app.route('/image_stylization/<model>', methods=['POST'])
def image_stylization(model):
  params_file = os.path.join(MODEL_DIR, model+'.params')
  if os.path.isfile(params_file):
    return jsonify({'ret_code': 2101, 'error': 'Unavailable style!'})

  img = request.values.get('image')
  try:
    img = img.replace(' ', '+')
    img = base64.decodestring(img)
  except:
    return jsonify({'ret_code':2102, 'error': 'Invalid base64 code!'})

  try:
    img = mx.image.imdecode(img, flag=1, to_rgb=0)
  except:
    return jsonify({'ret_code': 2103, 'error': 'Invalid image format!'})

  img = nstyle_forward.image_stylizing(img, params_file)
  img = np.roll(img, 1, axis=-1)
  img = Image.fromarray(img)

  return _send_image(img, 'jpg')

def _send_image(pil_img, format):
  img_io = StringIO()
  pil_img.save(img_io)
  img_io.seek(0)
  return send_file(img_io, cache_timeout=0)


if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0')