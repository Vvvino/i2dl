import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialise h as 0 if these values are not given.                          #
        ############################################################################

        self.hidden_size = hidden_size

        self.activation = activation

        self.lin_x = nn.Linear(input_size, hidden_size)
        self.lin_h = nn.Linear(hidden_size, hidden_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vector of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################

        hidden_size = self.hidden_size
        seq_len, batch_size, input_size = x.shape

        if h is None:
            h = torch.zeros((1, batch_size, hidden_size))

        activation = None

        if self.activation == "tanh":
            activation = torch.tanh
        if self.activation == "relu":
            activation = torch.relu

        x = self.lin_x(x)
        h = self.lin_h(h)

        h_seq.append(activation(h.sum(0)+x[0]))

        for t in range(1, seq_len):
            h_seq.append(activation(self.lin_h(h_seq[t-1])+x[t]))

        h = h_seq[-1]

        h_seq = torch.stack(h_seq)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq, h

    
class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
    ############################################################################
    # TODO: Build a one layer LSTM with an activation with the attributes      #
    # defined above and a forward function below. Use the nn.Linear() function #
    # as your linear layers.                                                   #
    # Initialse h and c as 0 if these values are not given.                    #
    ############################################################################

        self.hidden_size = hidden_size

        self.lin_f_x = nn.Linear(input_size, hidden_size)
        self.lin_f_h = nn.Linear(hidden_size, hidden_size)

        self.lin_i_x = nn.Linear(input_size, hidden_size)
        self.lin_i_h = nn.Linear(hidden_size, hidden_size)

        self.lin_o_x = nn.Linear(input_size, hidden_size)
        self.lin_o_h = nn.Linear(hidden_size, hidden_size)

        self.lin_c_x = nn.Linear(input_size, hidden_size)
        self.lin_c_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """

        h_seq = []
        c_seq = []

        hidden_size = self.hidden_size
        seq_len, batch_size, input_size = x.shape

        if h is None:
            h = torch.zeros((1, batch_size, hidden_size))
        if c is None:
            c = torch.zeros((1, batch_size, hidden_size))

        h = h.sum(0)
        c = c.sum(0)

        sigm = torch.sigmoid
        tanh = torch.tanh

        f_x = self.lin_f_x(x)
        f_h = self.lin_f_h(h)

        i_x = self.lin_i_x(x)
        i_h = self.lin_i_h(h)

        o_x = self.lin_o_x(x)
        o_h = self.lin_o_h(h)

        c_x = self.lin_c_x(x)
        c_h = self.lin_c_h(h)

        f = sigm(f_x[0] + f_h)
        i = sigm(i_x[0] + i_h)
        o = sigm(o_x[0] + o_h)

        c_seq.append(f*c + i*tanh(c_x[0] + c_h[0]))

        h_seq.append(o*tanh(c_seq[-1]))

        for t in range(1, seq_len):
            f_h = self.lin_f_h(h_seq[-1])
            f_t = sigm(f_x[t] + f_h)

            i_h = self.lin_i_h(h_seq[-1])
            i_t = sigm(i_x[t] + i_h)

            o_h = self.lin_o_h(h_seq[-1])
            o_t = sigm(o_x[t] + o_h)

            c_h = self.lin_c_h(h_seq[-1])
            c_seq.append(f_t*c_seq[-1] + i_t*tanh(c_x[t] + c_h))
            h_seq.append(o_t*tanh(c_seq[-1]))

        h = h_seq[-1]
        c = c_seq[-1]

        h_seq = torch.stack(h_seq)
        c_seq = torch.stack(c_seq)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################

        self.RNN = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        self.dense = nn.Linear(hidden_size, classes)
       
    def forward(self, x):

        x = self.RNN(x)[1]
        x = self.dense(x)[0]

        return x

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128, num_layers=1):
        super(LSTM_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a LSTM classifier                                           #
    ############################################################################

        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.dense = nn.Linear(hidden_size, classes)
    
    def forward(self, x):

        x = self.LSTM(x)[1][0]
        x = self.dense(x)[0]

        return x
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)