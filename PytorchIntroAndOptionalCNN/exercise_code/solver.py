from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
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
        iter_per_epochs = len(train_loader)
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

        train_data_size = len(train_loader)
        valid_data_size = len(val_loader)

        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))

            # Set to training mode
            model.train()

            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0

            valid_loss = 0.0
            valid_acc = 0.0

            for i, data in enumerate(train_loader):
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Clean existing gradients
                optim.zero_grad()

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss_criterion = torch.nn.CrossEntropyLoss()
                loss = loss_criterion(outputs, labels.long())

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optim.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)

                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions).long())

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                print(
                    "Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                model.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(val_loader, 0):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass - compute outputs on input data using the model
                    outputs = model(inputs)

                    # Compute loss
                    loss_criterion = torch.nn.CrossEntropyLoss()
                    loss = loss_criterion(outputs, labels.long())

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions).long())

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j,
                                                                                                               loss.item(),
                                                                                                               acc.item()))

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / float(train_data_size)

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / float(valid_data_size)

            self.train_loss_history.append(avg_train_loss)
            self.train_acc_history.append(avg_train_acc)
            self.val_loss_history.append(avg_valid_loss)
            self.val_acc_history.append(avg_valid_acc)

            print(
                "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                    epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
