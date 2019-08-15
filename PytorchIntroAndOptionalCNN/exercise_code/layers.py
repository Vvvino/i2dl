import numpy as np


def conv_single_step(x_slice, W, b):
    """
    Apply one filter defined by parameters W on a single slice (x_slice) of the output activation
    of the previous layer.

    Input:
    - x_slice: slice of input data of shape (C, HH, WW)
    - W: Weight parameters contained in a window - matrix of shape (C, HH, WW)
    - b: Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    - out: a scalar value, result of convolving the sliding window (W, b) on a slice of the input data
    """

    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(x_slice, W)
    # Sum over all entries of the volume s.
    out = np.sum(s)
    # Add bias b to out. Cast b to a float() so that out results in a scalar value.
    out += b

    return out


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    source: https://hackernoon.com/convnet-from-scratch-just-lovely-numpy-forward-pass-part-1-6d3a0776f90a

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (X, W, b, conv_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    s = conv_param['stride']
    pad = conv_param['pad']

    n_H = 1 + int((H + 2 * pad - HH) / s)
    n_W = 1 + int((W + 2 * pad - WW) / s)

    out = np.zeros(N, F, n_H, n_W)

    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)

    for i in range(N):  # loop over the batch of training examples
        for f in range(F):  # loop over channels (= #filters) of the output volume
            for h in range(n_H):  # loop over vertical axis of the output volume
                for wi in range(n_W):  # loop over horizontal axis of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * s
                    vert_end = vert_start + HH
                    horiz_start = wi * s
                    horiz_end = horiz_start + WW

                    # Using the corners to define the (3D) slice of a_prev_pad (See Hint above the cell).
                    x_slice = x_pad[i, :, vert_start:vert_end, horiz_start:horiz_end]

                    # Convolve the (3D) slice with the correct filter w and bias b, to get back one output neuron.
                    out[i, f, h, wi] = conv_single_step(x_slice, w[f], b[f])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    source: https://cthorey.github.io/backprop_conv/

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, n_H, n_W = dout.shape

    s = conv_param['stride']
    pad = conv_param['pad']

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    db = np.zeros_like(b)
    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])

    dw = np.zeros_like(w)
    for f in range(F):
        for c in range(C):
            for h in range(HH):
                for wi in range(WW):
                    # Find the corners of the current "slice"
                    vert_start = h * s
                    vert_end = vert_start + n_H
                    horiz_start = wi * s
                    horiz_end = horiz_start + n_W

                    x_slice = x_pad[:, c, vert_start:vert_end, horiz_start:horiz_end]

                    dw[f, c, h, wi] = np.sum(dout[:, f, :, :] * x_slice)

    dx = np.zeros_like(x)
    for i in range(N):
        for c in range(C):
            for h in range(H):
                for wi in range(W):
                    for f in range(F):
                        for k in range(n_H):
                            for l in range(n_W):
                                for p in range(HH):
                                    for q in range(WW):
                                        if (p + k * s == h + pad) & (q + s * l == wi + pad):
                                            dx[i, c, h, wi] += dout[i, f, k, l] * w[f, c, p, q]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, maxIdx, pool_param) for the backward pass with maxIdx, of shape (N, C, H, W, 2)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################

    N, C, H, W = x.shape

    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    s = pool_param['stride']

    n_H = int((H - pH) / s) + 1
    n_W = int((W - pW) / s) + 1

    maxIdx = np.zeros((N, C, H, W, 2))
    out = np.zeros((N, C, n_H, n_W))

    for i in range(N):  # loop over the batch of training examples
        for c in range(C):
            for h in range(n_H):  # loop over vertical axis of the output volume
                for wi in range(n_W):  # loop over horizontal axis of the output volume
                    # Find the corners of the current "slice"
                    vert_start = h * s
                    vert_end = vert_start + pH
                    horiz_start = wi * s
                    horiz_end = horiz_start + pW

                    x_slice = x[i, c, vert_start:vert_end, horiz_start:horiz_end]

                    arg_max = np.argmax(x_slice)

                    maxIdx[i, c, vert_start:vert_end, horiz_start:horiz_end, 0] = arg_max // pW + vert_start
                    maxIdx[i, c, vert_start:vert_end, horiz_start:horiz_end, 1] = arg_max % pW + horiz_start

                    out[i, c, h, wi] = np.max(x_slice)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, maxIdx, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    
    x, maxIdx, pool_param = cache

    N, C, H, W = x.shape
    N, C, n_H, n_W = dout.shape

    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    s = pool_param['stride']

    dx = np.zeros_like(x)
    for i in range(N):
        for c in range(C):
            for h in range(H):
                for wi in range(W):
                    for k in range(n_H):
                        for l in range(n_W):
                            vert_start = k * s
                            vert_end = vert_start + pH
                            horiz_start = l * s
                            horiz_end = horiz_start + pW

                            if (h == maxIdx[i, c, h, wi, 0]) & (wi == maxIdx[i, c, h, wi, 1]) & \
                                    (h in range(vert_start,vert_end)) & (wi in range(horiz_start,horiz_end)):
                                dx[i, c, h, wi] += dout[i, c, k, l]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    
    pass

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    
    pass 

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N, D = dout.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  
  pass 

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  pass 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
