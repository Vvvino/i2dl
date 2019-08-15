import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################

        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)

        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 30)

        # nn.init.uniform_(self.conv1.weight)
        # nn.init.uniform_(self.conv2.weight)
        # nn.init.uniform_(self.conv3.weight)
        # nn.init.uniform_(self.conv4.weight)

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

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
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################

        x = F.max_pool2d(F.elu(self.conv1(x)), 2)
        x = F.dropout(x, p=0.1)
        x = F.max_pool2d(F.elu(self.conv2(x)), 2)
        x = F.dropout(x, p=0.2)
        x = F.max_pool2d(F.elu(self.conv3(x)), 2)
        x = F.dropout(x, p=0.3)
        x = F.max_pool2d(F.elu(self.conv4(x)), 2)
        x = F.dropout(x, p=0.4)

        x = x.view(-1, self.num_flat_features(x))

        x = F.dropout(F.elu(self.fc1(x)), p=0.5)
        x = F.dropout(self.fc2(x), p=0.6)

        x = self.fc3(x)

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
