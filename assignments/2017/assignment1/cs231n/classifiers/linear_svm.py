import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    # !!! added !!!
    # number of incorrect predictions: not satisfying margin
    num_incorrect = 0
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        num_incorrect+=1
        # !!! added !!!
        # replacing dW row with normalized derivative mentioned in the course notes
        # incorrect class case
        dW[:,j]+=X[i]
    # !!! added !!!
    # replacing dW row with normalized derivative mentioned in the course notes
    # correct class case
    dW[:, y[i]] += -num_incorrect * X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # !!! added !!!
  #also needs to be normalized according to formula
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # !!! added !!!
  # derivative of regularization
  dW+= 2* reg * W
  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_scores = scores[xrange(num_train),y[xrange(num_train)]]

  margins=scores.T - correct_class_scores + 1  # note delta = 1
  margins[y[xrange(num_train)],xrange(num_train)]=0 #margins are not considered for correct classes
  margins=np.maximum(0,margins) # discarding satisfied margins


  loss = np.sum(margins)
  loss /= num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #margins were clamped to 0 from bottom before
  #first clamp margin values to 1 from top
  #then all positive values will be ceiled to 1
  #giving identity coefs for incorrect class gradients and zeroing all other cases
  margins_gradient_coefficients=np.ceil(np.minimum(1,margins))

  #1s indicate incorrect margin penalties so we sum them up for each example
  sums = np.sum(margins_gradient_coefficients, axis=0)

  #for each correct class in each example alter value from 0 to -sums
  #which is the correct class' gradient coefficient
  margins_gradient_coefficients[y[xrange(num_train)], xrange(num_train)] = -sums

  dW = dW.T
  #perform following multiplication to formulate correct gradient for all cases
  #respecting both gradients introduced in the course notes
  dW[xrange(num_classes)] += margins_gradient_coefficients.dot(X)

  dW=dW.T
  dW/=num_train
  dW+= 2* reg * W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
