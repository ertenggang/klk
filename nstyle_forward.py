import mxnet as mx
import glob
import time
import resnet_fast_style
import image_process

def get_executor(pretrained, input_sym, input_size, ctx):
  # make executor
  if isinstance(input_size, list):
    input_size = dict(input_size)
  arg_shapes, output_shapes, aux_shapes = input_sym.infer_shape(**input_size)
  arg_names = input_sym.list_arguments()
  arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
  # print arg_dict
  # print pretrained
  # init with pretrained weight
  input_names = input_size.keys()
  for name in arg_names:
    if name in input_names:
      continue
    # key = "arg:" + name
    key = name
    if key in pretrained:
      pretrained[key].copyto(arg_dict[name])
    else:
      print("Skip argument %s" % name)
  executor = input_sym.bind(ctx=ctx, args=arg_dict, grad_req="write")
  # return ConvExecutor(executor=executor,
  #                     outputs=executor.outputs,
  #                     arg_dict=arg_dict)
  return executor

def image_stylizing(img, params_file):
  net = resnet_fast_style.style_transfer_net()
  pretrained_model = mx.nd.load(params_file)
  img = image_process.preprocessing_image(img)
  img = mx.nd.array(img)
  input_size = {'content': img.shape}
  # print input_size
  ex = get_executor(pretrained_model, net, input_size, mx.cpu())
  img.copyto(ex.arg_dict['content'])
  ex.forward()
  out_img = ex.outputs[0].asnumpy()
  out_img = image_process.postprocessing_image(out_img)
  out_img = out_img[0]
  return out_img

def test():
  # test_path = 'train_img/content/'
  # test_imgs = glob.glob(test_path+'*.jpg')
  max_edge = 256
  test_imgs = ['test_img/tree.jpg']
  params_file = 'pretrained/resnet_starry_e4000.params'
  for f in test_imgs:
    t1 = time.time()
    img = open(f, 'rb').read()
    img = mx.image.imdecode(img, flag=1, to_rgb=0)
    print img.shape
    me = max(img.shape)
    print img.shape
    scale = float(max_edge)/me
    print  img.shape[0]*scale, img.shape[1]*scale
    img = mx.image.imresize(img, int(img.shape[0]*scale), int(img.shape[1]*scale))
    out_img = image_stylizing(img, params_file)
    t2 = time.time()
    print 'time: %f'%(t2-t1)

if __name__ == '__main__':
  test()