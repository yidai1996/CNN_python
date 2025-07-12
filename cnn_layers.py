from builtins import range
import numpy as np
import math

def conv_forward(x, w):

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    out = np.zeros((N, F, H_prime, W_prime))

    # reshape filters for vectorized dot
    w_col = w.reshape(F, -1)
    for j in range(H_prime):
        for i in range(W_prime):
            x_slice = x[:, :, j:j+HH, i:i+WW]  # N x C x HH x WW
            x_col = x_slice.reshape(N, -1)     # N x (C*HH*WW)
            # compute output: N x F = (N x M) dot (M x F)
            out[:, :, j, i] = x_col.dot(w_col.T)

    cache = (x, w)
    return out, cache


def conv_backward(dout, cache,stride=1):
    x, w = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = (H - HH) // stride + 1
    W_out = (W - WW) // stride + 1

    # 1) im2col: unfold x into columns
    cols = []
    for j in range(0, H_out * stride, stride):
        for i in range(0, W_out * stride, stride):
            patch = x[:, :, j:j+HH, i:i+WW]  # N x C x HH x WW
            cols.append(patch.reshape(N, -1))
    x_cols = np.stack(cols, axis=1).reshape(-1, C * HH * WW)  # (N*H_out*W_out, C*HH*WW)

    # 2) reshape dout to 2D
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, F)        # (N*H_out*W_out, F)

    # 3) compute dw in one matrix multiplication
    dw_flat = dout_flat.T.dot(x_cols)                            # (F, C*HH*WW)
    dw = dw_flat.reshape(F, C, HH, WW)

    # 4) compute dx columns
    w_flat = w.reshape(F, -1)                                    # (F, C*HH*WW)
    dx_cols = dout_flat.dot(w_flat)                              # (N*H_out*W_out, C*HH*WW)
    dx_cols = dx_cols.reshape(N, H_out * W_out, C * HH * WW)

    # 5) col2im: fold columns back into dx
    dx = np.zeros_like(x)
    idx = 0
    for j in range(0, H_out * stride, stride):
        for i in range(0, W_out * stride, stride):
            col = dx_cols[:, idx, :].reshape(N, C, HH, WW)
            dx[:, :, j:j+HH, i:i+WW] += col
            idx += 1

    return dx, dw



def max_pool_forward(x, pool_param):
    out = None
    (N, C, H, W) = x.shape

    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    S = (N, C, math.floor(H_out), math.floor(W_out))
    out = np.zeros(S)

    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            out[:, :, x1, y] = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    dx = None

    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    dx = np.zeros(x.shape)

    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            max_element = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
            max_element = max_element[:,:,np.newaxis]
            max_element = np.repeat(max_element, p_H, axis=2)
            max_element = max_element[:,:,:,np.newaxis]
            max_element = np.repeat(max_element, p_W, axis=3)
            temp = np.zeros(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
            temp = (x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
            tmp_dout = dout[:,:,x1,y]
            tmp_dout = tmp_dout[:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_H, axis=2)
            tmp_dout = tmp_dout[:,:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_W, axis=3)
            dx[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += tmp_dout * temp
    return dx
