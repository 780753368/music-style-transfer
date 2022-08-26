import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import torch
##from torch.functional import Tensor
from ops import *
from utils import *
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import PatchEmbed, Block
from utils import *

##weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer_fully = None
weight_regularizer = None
img_size=64
patch_size=16
in_chans=1
embed_dim=168
depth=24
num_heads=16
decoder_embed_dim=168
decoder_depth=8
decoder_num_heads=16
mlp_ratio=4
norm_layer=nn.LayerNorm
norm_pix_loss=False
patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
num_patches = 64
N=16
L=64
H=84
D=2
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=7, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)


def deconv2d(input_, output_dim, ks=7, s=2, stddev=0.02, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def relu(tensor_in):
    if tensor_in is not None:
        return tf.nn.relu(tensor_in)
    else:
        return tensor_in


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def to_binary_tf(bar_or_track_bar, threshold=0.0, track_mode=False, melody=False):
    """Return the binarize tensor of the input tensor (be careful of the channel order!)"""
    if track_mode:
        # melody track
        if melody:
            melody_is_max = tf.equal(bar_or_track_bar, tf.reduce_max(bar_or_track_bar, axis=2, keep_dims=True))
            melody_pass_threshold = (bar_or_track_bar > threshold)
            out_tensor = tf.logical_and(melody_is_max, melody_pass_threshold)
        # non-melody track
        else:
            out_tensor = (bar_or_track_bar > threshold)
        return out_tensor
    else:
        if len(bar_or_track_bar.get_shape()) == 4:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0], [-1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 1], [-1, -1, -1, -1])
        elif len(bar_or_track_bar.get_shape()) == 5:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 1], [-1, -1, -1, -1, -1])
        # melody track
        melody_is_max = tf.equal(melody_track, tf.reduce_max(melody_track, axis=2, keep_dims=True))
        melody_pass_threshold = (melody_track > threshold)
        out_tensor_melody = tf.logical_and(melody_is_max, melody_pass_threshold)
        # other tracks
        out_tensor_others = (other_tracks > threshold)
        if len(bar_or_track_bar.get_shape()) == 4:
            return tf.concat([out_tensor_melody, out_tensor_others], 3)
        elif len(bar_or_track_bar.get_shape()) == 5:
            return tf.concat([out_tensor_melody, out_tensor_others], 4)


def to_chroma_tf(bar_or_track_bar, is_normalize=True):
    """Return the chroma tensor of the input tensor"""
    out_shape = tf.stack([tf.shape(bar_or_track_bar)[0], bar_or_track_bar.get_shape()[1], 12, 7,
                         bar_or_track_bar.get_shape()[3]])
    chroma = tf.reduce_sum(tf.reshape(tf.cast(bar_or_track_bar, tf.float32), out_shape), axis=3)
    if is_normalize:
        chroma_max = tf.reduce_max(chroma, axis=(1, 2, 3), keep_dims=True)
        chroma_min = tf.reduce_min(chroma, axis=(1, 2, 3), keep_dims=True)
        return tf.truediv(chroma - chroma_min, (chroma_max - chroma_min + 1e-15))
    else:
        return chroma


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def conv2d_musegan(tensor_in, out_channels, kernels, strides, stddev=0.02, name='conv2d', reuse=None, padding='VALID'):
    """
    Apply a 2D convolution layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        out_channels (int): The number of output channels.
        kernels (list of int): The size of the kernel. [kernel_height, kernel_width]
        strides (list of int): The stride of the sliding window. [stride_height, stride_width]
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        name (str): The tenorflow variable scope. Defaults to 'conv2d'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            weights = tf.get_variable('weights', kernels+[tensor_in.get_shape()[-1], out_channels],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(tensor_in, weights, strides=[1]+strides+[1], padding=padding)

            out_shape = tf.stack([tf.shape(tensor_in)[0]]+list(conv.get_shape()[1:]))

            return tf.reshape(tf.nn.bias_add(conv, biases), out_shape)


def deconv2d_musegan(tensor_in, out_shape, out_channels, kernels, strides, stddev=0.02, name='transconv2d', reuse=None,
                padding='VALID'):
    """
    Apply a 2D transposed convolution layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        out_shape (list of int): The output shape. [height, width]
        out_channels (int): The number of output channels.
        kernels (list of int): The size of the kernel.[kernel_height, kernel_width]
        strides (list of int): The stride of the sliding window. [stride_height, stride_width]
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        name (str): The tenorflow variable scope. Defaults to 'transconv2d'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            # filter : [height, width, output_channels, in_channels]
            weights = tf.get_variable('weights', kernels+[out_channels, tensor_in.get_shape()[-1]],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            output_shape = tf.stack([tf.shape(tensor_in)[0]]+out_shape+[out_channels])

            try:
                conv_transpose = tf.nn.conv2d_transpose(tensor_in, weights, output_shape=output_shape,
                                                        strides=[1]+strides+[1], padding=padding)
            except AttributeError:  # Support for verisons of TensorFlow before 0.7.0
                conv_transpose = tf.nn.deconv2d(tensor_in, weights, output_shape=output_shape, strides=[1]+strides+[1],
                                                padding=padding)

            return tf.reshape(tf.nn.bias_add(conv_transpose, biases), output_shape)
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False,name='conv_0'):
    with tf.variable_scope(name):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x
def init_down_resblock(x_init, channels, use_bias=True, sn=False, name='resblock'):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = down_sample(x)

        with tf.variable_scope('shortcut'):
            x_init = down_sample(x_init)
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x + x_init
def down_resblock(x_init, channels, to_down=True, use_bias=True, sn=False, name='resblock'):
    with tf.variable_scope(name):
        init_channel = x_init.shape.as_list()[-1]
        with tf.variable_scope('res1'):
            x = lrelu(x_init, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = lrelu(x, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

            if to_down :
                x = down_sample(x)

        if to_down or init_channel != channels :
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)
                if to_down :
                    x_init = down_sample(x_init)


        return x + x_init
def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])

    return gsp
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def up_resblock(x_init, channels, use_bias=True, is_training=True, sn=False, name='resblock'):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            x = batch_norm(x_init, is_training)
            x = relu(x)
            x = up_sample(x, scale_factor=2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False,sn=sn)

        with tf.variable_scope('res2'):
            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias,sn=sn)

        with tf.variable_scope('shortcut'):
            x_init = up_sample(x_init, scale_factor=2)
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False,sn=sn)

        return x + x_init
def flatten(x) :
    return tf.layers.flatten(x)
def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_sample(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def max_pooling(x) :
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def fullyconnected(x, units, use_bias=True, sn=False, name='linear'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def batch_norm(x, is_training=True, name='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=name)
def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    imgs=np.array(imgs)
    ##imgs = torch.einsum('nhwc->nchw', imgs)
    ##p = patch_embed.patch_size[0]
    ##print('p:',p)
    ##assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    ##h = w = imgs.shape[2] // p
    ##x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    ##print(x.shape)
    ##x = torch.einsum('nchpwq->nhwpqc', x)
    ##print(x.shape)
    n,l,h,d=imgs.shape
  
    x = imgs.reshape(imgs.shape[0], l, h*d)
    ##print(x.shape)
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_embed.patch_size[0]
    ##h = w = int(x.shape[1]**.5)
    ##assert h * w == x.shape[1]
    x = x.reshape(x.shape[0],L,H,D)
    ##x = torch.einsum('nhwpqc->nchpwq', x)
    ##imgs = x.reshape(shape=(x.shape[0], L, H , D))
    ##print('out:',imgs,imgs.shape)
    ##imgs = torch.einsum('nchw->nhwc', imgs)
    
    return x

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L)  # noise in [0, 1]
    ##print('noise',noise,noise.shape)
        # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ##print('ids_shuffle:',ids_shuffle,ids_shuffle.shape)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ##print('ids_restore:',ids_restore,ids_restore.shape)
        # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ##print('ids_keep:',ids_keep,ids_keep.shape)
    ##print('index:',ids_keep.unsqueeze(-1).repeat(1, 1, D),ids_keep.unsqueeze(-1).repeat(1, 1, D).shape)
    x=torch.tensor(x)
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 168))
        # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L])
    mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
def to_numpy(x):
    with tf.Session() as sess:
        data_numpy = x.eval()
    return data_numpy
    


    