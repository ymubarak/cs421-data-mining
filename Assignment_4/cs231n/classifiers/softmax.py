import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # for i in range(len(y)):
  #   logits = X[i].dot(W)
  #   exps = np.exp(z)
  #   sum_of_exps = np.sum(exps)
  #   sm = exps/sum_of_exps

  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
      logits = X[i].dot(W)
      logits -= np.max(logits)
      exps = np.exp(logits)
      exp_sum = np.sum(exps)
      probs = exps/exp_sum
      h = probs[y[i]]
      loss -= np.log(h)
      #compute gradient
      for j in range(num_classes):
          dW[:,j] += (probs[j] - (j==y[i])) * X[i]


  loss /= num_train
  dW /= num_train

  # regulariztion
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = len(y)
  logits = X.dot(W)
  logits -= np.max(logits, axis=1, keepdims=1)
  exps = np.exp(logits)
  probs = exps/np.sum(exps,axis=1, keepdims=1)

  # 1. compute softmax loss
  loss = -np.sum(np.log(probs[np.arange(num_train),y]))
  loss /= num_train
  # regularization
  loss += 0.5*reg*np.sum(W*W)

  # 2. Compute the gradient
  probs[np.arange(num_train),y] -= 1
  dW = X.T.dot(probs)
  dW /= num_train
  # regularization
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

