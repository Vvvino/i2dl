from random import shuffle
import numpy as np
from distutils.version import LooseVersion

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='mean', ignore_index=-1)
    #if size_average:
    #    loss /= mask.data.sum()
    return loss


class Solver(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################



        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))

            # Loss and Accuracy within the epoch
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = []
            valid_acc = []

            model.train()

            for data in train_loader:
                input, target = data

                input = input.to(device)
                target = target.to(device)

                input = torch.autograd.Variable(input)
                target = torch.autograd.Variable(target)

                # Clean existing gradients
                optim.zero_grad()

                # Forward pass - compute outputs on input data using the model
                output = model(input)

                # Compute loss
                loss = self.loss_func(output, target)
                # loss = cross_entropy2d(output, target)

                loss_data = loss.item()
                loss_data = loss_data / len(input)

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optim.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss_data

                _, preds = torch.max(output, 1)
                targets_mask = target >= 0
                train_acc.append(np.mean((preds == target)[targets_mask].data.cpu().numpy()))

                #print("Batch number: {:03d}, Training: Loss: {:.4f}".format(i, loss_data))

            with torch.no_grad():

                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data in val_loader:
                    input, target = data

                    input = torch.autograd.Variable(input)
                    target = torch.autograd.Variable(target)

                    input = input.to(device)
                    target = target.to(device)

                    # Forward pass - compute outputs on input data using the model
                    output = model(input)

                    # Compute loss
                    loss = self.loss_func(output, target)
                    # loss = cross_entropy2d(output, target)

                    loss_data = loss.data.item() / len(input)

                    valid_loss += loss_data

                    _, preds = torch.max(output, 1)
                    targets_mask = target >= 0
                    valid_acc.append(np.mean((preds == target)[targets_mask].data.cpu().numpy()))

                    #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}".format(j, loss.item()))

            avg_train_loss = train_loss / len(train_loader)
            avg_valid_loss = valid_loss / len(val_loader)

            avg_train_acc = np.mean(train_acc)
            avg_valid_acc = np.mean(valid_acc)

            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_valid_loss)

            self.train_acc_history.append(avg_train_acc)
            self.val_acc_history.append(avg_valid_acc)

            print("Training loss: {:.4f} \tValidation loss : {:.4f}".format(avg_train_loss, avg_valid_loss))
            print("Training Acc: {:.4f} \tValidation Acc : {:.4f}".format(avg_train_acc, avg_valid_acc))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
