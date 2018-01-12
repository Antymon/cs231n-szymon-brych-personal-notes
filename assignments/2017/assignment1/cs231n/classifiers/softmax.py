import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #num_dim = X.shape[1]

  for i in xrange(num_train):
    correct_class_index = y[i]

    scores = X[i].dot(W)
    scores -= np.max(scores) #subtracting constant for the sake of numeric safety; this operation is meaningless in mathematical sense
    exp_scores = np.exp(scores)
    normalized_scores = exp_scores/np.sum(exp_scores)

    normalized_correct_class_score = normalized_scores[correct_class_index]
    loss -= np.log(normalized_correct_class_score)

    #gradient in relation to score
    dscores = normalized_scores
    dscores[correct_class_index]-=1

    #gradient in relation to weight
    for c in xrange(num_classes):
      # super slow, fully looped:
      # for n in xrange(num_dim):
      #   dW[n,c]+=dscores[c]*X[i,n]
      # one level of vectorization:
      dW[:,c]+=dscores[c]*X[i,:]

  loss /= num_train

  dW /= num_train
  loss += reg * np.sum(W * W)
  dW+= 2* reg * W

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
  num_train = X.shape[0]

  scores = X.dot(W)
  #subtracting constant for the sake of numeric safety
  #this operation is meaningless result-wise aside from numeric concerns
  scores -= np.max(scores,1,keepdims=True)
  exp_scores = np.exp(scores)

  #if keepdims is off tiling would be needed
  #as matrix mul and div are component-wise, not matrix operations
  sums = np.sum(exp_scores,1,keepdims=True)
  normalized_scores = exp_scores/sums

  normalized_correct_class_score = normalized_scores[np.arange(num_train),y]
  loss -= np.sum(np.log(normalized_correct_class_score)) / num_train
  loss += reg * np.sum(W * W)

  #gradient in relation to score
  dscores = normalized_scores
  dscores[np.arange(num_train),y]-=1

  #gradient in relation to weight
  dW=np.transpose(X).dot(dscores)
  dW /= num_train

  #regularization gradient
  dW+= 2* reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW