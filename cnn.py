import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        self.params = {}
        self.dtype = dtype
        C, H, W = input_dim
        F1, F2 = num_filters_1, num_filters_2
        K = filter_size
        # Initialization scales
        # Conv1: k = 1/(C * K^2)
        scale1 = np.sqrt(1.0 / (C * K * K))
        # Conv2: k = 1/(F1 * K^2)
        scale2 = np.sqrt(1.0 / (F1 * K * K))
        # After two 2x2 pools, spatial dims are H/4 x W/4
        H2 = H // 4
        W2 = W // 4
        D_flat = F2 * H2 * W2
        # FC1: k = 1/D_flat
        scale3 = np.sqrt(1.0 / D_flat)
        # FC2: k = 1/hidden_dim
        scale4 = np.sqrt(1.0 / hidden_dim)

        # Weights initialization
        self.params['W1'] = np.random.uniform(-scale1, scale1, size=(F1, C, K, K))
        self.params['W2'] = np.random.uniform(-scale2, scale2, size=(F2, F1, K, K))
        self.params['W3'] = np.random.uniform(-scale3, scale3, size=(D_flat, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = np.random.uniform(-scale4, scale4, size=(hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)

        # Cast to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        W1, W2 = self.params['W1'], self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        # Conv params (preserve spatial size)
        pad = (self.params['W1'].shape[2] - 1) // 2
        conv_param = {'stride': 1, 'pad': pad}
        # Pool params
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # Forward pass
        # 1. Conv -> ReLU -> Pool
        conv1, cache1 = conv_forward(X, W1, conv_param)
        relu1, cache_relu1 = relu_forward(conv1)
        pool1, cache_pool1 = max_pool_forward(relu1, pool_param)
        # 2. Conv -> ReLU -> Pool
        conv2, cache2 = conv_forward(pool1, W2, conv_param)
        relu2, cache_relu2 = relu_forward(conv2)
        pool2, cache_pool2 = max_pool_forward(relu2, pool_param)
        # 3. Flatten -> FC -> ReLU
        N, F2, H2, W2 = pool2.shape
        flat = pool2.reshape(N, -1)
        fc3, cache_fc3 = fc_forward(flat, W3, b3)
        relu3, cache_relu3 = relu_forward(fc3)
        # 4. FC -> scores
        scores, cache_fc4 = fc_forward(relu3, W4, b4)

        if y is None:
            return scores

        # Compute loss and gradients
        loss, dscores = softmax_loss(scores, y)
        grads = {}
        # Backprop FC2
        drelu3, dW4, db4 = fc_backward(dscores, cache_fc4)
        grads['W4'] = dW4
        grads['b4'] = db4
        # Backprop ReLU3, FC3
        dfc3 = relu_backward(drelu3, cache_relu3)
        dflat, dW3, db3 = fc_backward(dfc3, cache_fc3)
        grads['W3'] = dW3
        grads['b3'] = db3
        # Reshape to conv shape
        dpool2 = dflat.reshape(pool2.shape)
        # Backprop pool2, relu2, conv2
        drelu2 = max_pool_backward(dpool2, cache_pool2)
        dconv2 = relu_backward(drelu2, cache_relu2)
        dpool1, dW2 = conv_backward(dconv2, cache2)
        grads['W2'] = dW2
        # Backprop pool1, relu1, conv1
        drelu1 = max_pool_backward(dpool1, cache_pool1)
        dconv1 = relu_backward(drelu1, cache_relu1)
        _, dW1 = conv_backward(dconv1, cache1)
        grads['W1'] = dW1

        return loss, grads
