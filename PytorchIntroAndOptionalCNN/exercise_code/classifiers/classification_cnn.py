"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################

        self.pool = pool
        self.dropout = dropout

        pad = (kernel_size - 1) // 2

        conv_h = (height + 2*pad - kernel_size) // stride_conv + 1
        conv_w = (width + 2*pad - kernel_size) // stride_conv + 1

        pool_H, pool_W = (conv_h - pool) // stride_pool + 1,  (conv_w - pool) // stride_pool + 1

        self.conv = nn.Conv2d(channels, num_filters, kernel_size, stride=stride_conv, padding=pad)

        self.fc1 = nn.Linear(num_filters * pool_H * pool_W, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        weight = self.conv.weight.clone()

        weight.mul_(weight_scale)

        self.conv.weight = torch.nn.Parameter(weight)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################

        pool = self.pool
        dropout = self.dropout

        x = F.max_pool2d(F.relu(self.conv(x)), (pool, pool))

        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(self.fc1(x), p=dropout)
        x = F.relu(x)
        x = self.fc2(x)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
