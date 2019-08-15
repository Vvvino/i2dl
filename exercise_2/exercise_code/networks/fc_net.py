import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


def affine_relu_batchnorm_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform with batch normalization
    followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta, bn_param: params for batch normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a_norm, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_norm)
    cache = (fc_cache, relu_cache, bn_cache)
    return out, cache


def affine_relu_batchnorm_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer with batch normalization
    """
    fc_cache, relu_cache, bn_cache = cache
    da = relu_backward(dout, relu_cache)
    da_norm, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(da_norm, fc_cache)
    return dx, dw, db, dgamma, dbeta


def relu_dropout_forward(x, dropout_param):
    """
    Convenience layer that perorms a dropout after a ReLU

    Inputs:
    - x: input to dropout
    - cache: input cache
    - dropout_param

    Returns a tuple of:
    - out: Output from the dropout
    - cache: Object to give to the backward pass
    """
    out, cache = dropout_forward(x, dropout_param)

    return out, cache


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure should be affine - relu - affine - softmax.
  
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, loss_function='softmax'):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        if loss_function != 'softmax':
            raise Exception('Wrong loss function')
        else:
            self.loss_function = 'softmax'

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_layer, hidden_cache = affine_relu_forward(X, W1, b1)

        scores, scores_cache = affine_forward(hidden_layer, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.reg * np.sum(np.square(W1))  # L2 Regularization for W1
        loss += 0.5 * self.reg * np.sum(np.square(W2))  # L2 Regularization for W2

        dhidden, dw2, db2 = affine_backward(dscores, scores_cache)

        grads['b2'] = db2

        grads['W2'] = dw2

        grads['W2'] += self.reg * W2

        dx, dw1, db1 = affine_relu_backward(dhidden, hidden_cache)

        grads['b1'] = db1

        grads['W1'] = dw1

        grads['W1'] += self.reg * W1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - loss function
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,
                 loss_function='softmax'):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.loss_function = loss_function
        if loss_function == 'softmax':
            self.chosen_loss_function = softmax_loss
        elif loss_function == 'l2':
            self.chosen_loss_function = l2_loss
        else:
            raise Exception('Wrong loss function')

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])

        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(hidden_dims[0])
            self.params['beta1'] = np.zeros(hidden_dims[0])

        for i in range(2, self.num_layers):
            self.params['W' + str(i)] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
            self.params['b' + str(i)] = np.zeros(hidden_dims[i-1])
            if self.use_batchnorm:
                self.params['gamma' + str(i)] = np.ones(hidden_dims[i-1])
                self.params['beta' + str(i)] = np.zeros(hidden_dims[i-1])

        self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        num_hidden = self.num_layers - 1

        hidden_layers = [None] * num_hidden
        hidden_caches = [None] * num_hidden
        dropout_caches = [None] * num_hidden

        if self.use_batchnorm:
            hidden_layers[0], hidden_caches[0] = affine_relu_batchnorm_forward(X, self.params['W1'], self.params['b1'],
                                                                               self.params['gamma1'],
                                                                               self.params['beta1'],
                                                                               self.bn_params[0])
        else:
            hidden_layers[0], hidden_caches[0] = affine_relu_forward(X, self.params['W1'], self.params['b1'])

        if self.use_dropout:
            hidden_layers[0], dropout_caches[0] = relu_dropout_forward(hidden_layers[0], self.dropout_param)

        for i in range(1, num_hidden):
            if self.use_batchnorm:
                hidden_layers[i], hidden_caches[i] = affine_relu_batchnorm_forward(hidden_layers[i - 1],
                                                                                   self.params['W' + str(i + 1)],
                                                                                   self.params['b' + str(i + 1)],
                                                                                   self.params['gamma' + str(i + 1)],
                                                                                   self.params['beta' + str(i + 1)],
                                                                                   self.bn_params[i])
            else:
                hidden_layers[i], hidden_caches[i] = affine_relu_forward(hidden_layers[i - 1],
                                                                         self.params['W' + str(i + 1)],
                                                                         self.params['b' + str(i + 1)])
            if self.use_dropout:
                hidden_layers[i], dropout_caches[i] = relu_dropout_forward(hidden_layers[i], self.dropout_param)

        scores, scores_cache = affine_forward(hidden_layers[-1], self.params['W'+str(num_hidden+1)],
                                              self.params['b'+str(num_hidden+1)])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # We compute loss and gradient of last layer
        # For another notebook, we will need to switch between different loss functions
        # By default we choose the softmax loss
        loss, dscores = self.chosen_loss_function(scores, y)
        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store#
        # the loss in the loss variable and gradients in the grads dictionary.#
        #                                                                     #
        #1_FullyConnectedNets.ipynb                                           #
        # Compute                                                             #
        # data loss using softmax, and make sure that grads[k] holds the      #
        # gradients for self.params[k]. Don't forget to add L2 regularization!#
        #                                                                     #
        # When using batch normalization, you don't need to regularize the    #
        # scale and shift parameters.                                         #
        #                                                                     #
        # NOTE: To ensure that your implementation matches ours and you pass  #
        # the automated tests, make sure that your L2 regularization includes #
        # a factor of 0.5 to simplify the expression for the gradient.        #
        #                                                                     #
        #######################################################################

        for i in range(self.num_layers):
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W'+str(i+1)]))

        dhidden = [None] * (num_hidden+1)
        dw = [None] * (num_hidden+1)
        db = [None] * (num_hidden+1)
        dgamma = [None] * num_hidden
        dbeta = [None] * num_hidden

        dhidden[-1], dw[-1], db[-1] = affine_backward(dscores, scores_cache)

        grads['b' + str(num_hidden + 1)] = db[-1]
        grads['W' + str(num_hidden + 1)] = dw[-1]
        grads['W' + str(num_hidden + 1)] += self.reg * self.params['W' + str(num_hidden + 1)]

        for i in reversed(range(num_hidden)):
            if self.use_dropout:
                dhidden[i + 1] = dropout_backward(dhidden[i + 1], dropout_caches[i])
            if self.use_batchnorm:
                dhidden[i], dw[i], db[i], dgamma[i], dbeta[i] = affine_relu_batchnorm_backward(dhidden[i + 1],
                                                                                               hidden_caches[i])
                grads['gamma' + str(i + 1)] = dgamma[i]
                grads['beta' + str(i + 1)] = dbeta[i]
            else:
                dhidden[i], dw[i], db[i] = affine_relu_backward(dhidden[i + 1], hidden_caches[i])

            grads['b' + str(i + 1)] = db[i]
            grads['W' + str(i + 1)] = dw[i]
            grads['W' + str(i + 1)] += self.reg * self.params['W' + str(i + 1)]


        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################


        return loss, grads
