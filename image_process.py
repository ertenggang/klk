import numpy as np
import mxnet as mx

def preprocessing_image(img):
  # print type(img)
  if isinstance(img, mx.ndarray.NDArray):
    img = img.asnumpy()
  img = img.astype('float32')
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2)

  img[0, :] -= 123.68
  img[1, :] -= 116.779
  img[2, :] -= 103.939

  shape = img.shape[1:]
  return np.resize(img, (1, 3, )+shape)

def postprocessing_image(img):
  # img = np.resize(img, (im.shape, img.shape[2], img.shape[3]))
  img[:, 0, :, :] += 123.68
  img[:, 1, :, :] += 116.779
  img[:, 2, :, :] += 103.939
  img = np.swapaxes(img, 2, 3)
  img = np.swapaxes(img, 1, 3)
  img = np.clip(img, 0, 255)
  return img.astype('uint8')