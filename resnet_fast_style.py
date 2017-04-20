import mxnet as mx

WORKSPACE = 1024

def resnet_block(input_sym, ksize, num_filter):
  pad_size = (0,0,0,0,ksize[0]//2, ksize[0]//2, ksize[1]//2, ksize[1]//2)
  net = mx.sym.Pad(data=input_sym, mode='edge', pad_width=pad_size)
  net = mx.sym.Convolution(data=net, kernel=ksize, num_filter=num_filter, no_bias=True, workspace=WORKSPACE)
  net = mx.sym.InstanceNorm(data=net)
  net = mx.sym.Activation(data=net, act_type='relu')
  net = mx.sym.Pad(data=net, mode='edge', pad_width=pad_size)
  net = mx.sym.Convolution(data=net, kernel=ksize, num_filter=num_filter, no_bias=True, workspace=WORKSPACE)
  net = mx.sym.InstanceNorm(data=net)
  net = net+input_sym
  return net

def conv_block(input_sym, ksize, stride, num_filter, act_type='relu'):
  pad = (ksize[0]//2, ksize[1]//2)
  net = mx.sym.Convolution(data=input_sym, kernel=ksize, num_filter=num_filter, pad=pad, stride=stride, no_bias=True, workspace=WORKSPACE)
  net = mx.sym.InstanceNorm(data=net)
  net = mx.sym.Activation(data=net, act_type=act_type)
  return net

def style_transfer_net():
  scale = 2
  net = mx.sym.Variable('content')
  net = conv_block(net, (9, 9), (1, 1), 32)
  net = conv_block(net, (3, 3), (scale, scale), 64)
  net = conv_block(net, (3, 3), (scale, scale), 128)
  net = resnet_block(net, ksize=(3, 3), num_filter=128)
  net = resnet_block(net, ksize=(3, 3), num_filter=128)
  net = resnet_block(net, ksize=(3, 3), num_filter=128)
  net = resnet_block(net, ksize=(3, 3), num_filter=128)
  net = resnet_block(net, ksize=(3, 3), num_filter=128)
  net = mx.sym.Deconvolution(
    data=net, 
    kernel=(2*scale, 2*scale), 
    stride=(scale, scale), 
    pad=(scale/2, scale/2), 
    num_filter=128, 
    no_bias=True, 
    workspace = WORKSPACE
    )
  net = conv_block(net, (3, 3), (1, 1), 64)
  net = mx.sym.Deconvolution(
    data=net, 
    kernel=(2*scale, 2*scale), 
    stride=(scale, scale), 
    pad=(scale/2, scale/2), 
    num_filter=128, 
    no_bias=True, 
    workspace = WORKSPACE
    )
  net = conv_block(net, (3, 3), (1, 1), 32)
  net = conv_block(net, (9, 9), (1, 1), 3, act_type='relu')
  net = net-127
  return net

