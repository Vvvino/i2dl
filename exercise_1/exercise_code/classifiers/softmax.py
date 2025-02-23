"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps,axis=1)[:,None]


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = X.shape[0]
    C = W.shape[1]
    
    for i in range(N):
        y_hat = np.dot(X[i],W)
        sum_exp = np.sum(np.exp(y_hat-np.max(y_hat)));
        
        for j in range(C):
            t = 1 if j==y[i] else 0            
            s_max = np.exp(y_hat[j]-np.max(y_hat))/sum_exp
            loss -= t*np.log(s_max)
            #dW.T[j] = X[i]*(s_max-t) / N
    
    loss /= N

    

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = y.shape[0]
    
    y_hat = np.dot(X,W) # Output layer
    
    #exps = np.sum(np.exp(y_hat-np.max(y_hat)),axis=1)
    #s_max = np.exp(y_hat-np.max(y_hat))
    #s_max = s_max / exps[:,None]
    #p = s_max[range(N),y]  
    #p = exp / exps
    
    p = stable_softmax(y_hat)

    log_likelihood = -np.log(p[range(N),y])
    loss = np.sum(log_likelihood) / N
    loss += reg*np.sum(np.square(W)) #L2 Regularization
    
    p[range(N),y] -= 1
    
    p /= N

    dW = np.dot(X.T, p)
    
    


    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW



class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = np.arange(1.7e-6, 2e-6, 1e-7)
    regularization_strengths = np.arange(1e2, 1e3, 1e2)

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    for i in learning_rates:
        for j in regularization_strengths:
            softmax = SoftmaxClassifier()
            
            softmax.train(X_train, y_train, learning_rate=i, reg=j,
                          num_iters=500, verbose=False)
            
            all_classifiers.append(softmax)
            
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            
            train_acc = np.mean(y_train == y_train_pred)
            val_acc = np.mean(y_val == y_val_pred)
                   
            results[(i,j)] = (train_acc, val_acc)
            
            if val_acc > best_val:
                best_val = val_acc
                best_softmax = softmax
    

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
