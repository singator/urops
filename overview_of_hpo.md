# Overview of Hyperparamter Optimization Course.

## Week 1

### C2W1L01
- Big data: no longer the 60/20/20 train/validate/test or 70/30 train/test
  anymore.
- Can be, e.g. 98/1/1 if you have enough data.

### C2W1L02
- Idea of consideration of bias and variance in context of the optimal (Bayes)
  error.
- Can have both high bias and variance.

### C2W1L03
- Solutions for reducing bias.
- Solutions for reducing variance.
- Deep learning: not necessary to have a tradeoff between bias and variance.
- Increasing one (through the adjustments mentioned) often has little to no
  detrimental effect on the other.

### C2W1L04, 05
- On L2 regularization, a solution to overfitting by penalizing models which
  assign great weight to parameters.
- Regularization more specifically in the context of neural networks.
- Considerations of how greater penalty leads to fitted models being more
  linear.

### C2W1L06, 07
- On Dropout regularization -- how we disable and train our model on different
  nodes for each training example.
- Specifically, on implementing inverted dropout.
- Considerations of how dropout shrinks weights.
- Considerations of keep probabilities for different layers.

### C2W1L08
- Other regularization techniques: data augmentation through manipulation of
  existing training examples.
- Early stopping as another technique.
- Introduction to orthogonalization: doing one thing at a time.

### C2W1L09
- On normalization, importance of consistency between train and test sets.
- How normalization speeds up gradient descent.

### C2W1L10, 11
- On vanishing, exploding gradients found in deep neural networks.
- Why they make gradient descent slow.
- Activation function-dependent solutions to this.

### C2W1L12+
- Gradient checking. *Seems unnecessary if we are to use tensorflow, which
  implements backpropagation automatically when an optimization function is
  specified.*
  
## Week 2

### C2W2L01
- Need for fast optimization algorithms.
- Explanation of mini-batch gradient descent and its implentation.
- Explanation of epochs with relation to mini-batch gradient descent.

### C2W2L02
- Possibility of non-monotocity of cost function over multiple epochs when using
  mini-batch gradient descent.
- Introduction to stochaistic gradient descent, and its noisiness.
- The reason why both stochaistic and batch gradient descent are slow.
- Guidlines for choosing a mini-batch size.
- Typical sizes: 64, 128, 256, 512, 1024.

### C2W2L03, 04
- Introduction to exponentially-weighted moving averages and their
  implementation. 

### C2W2L04
- Role of exponentially-weighted moving averages in optimization.

### C2W2L05
- Bias correction of exponentially-weighted moving averages.

### C2W2L06
- Gradient descent with momentum -- uses exponentially-weighted moving averages.
- Why this version is faster than vanilla -- smooths out oscillations.

### C2W2L07
- An introduction to the RMSProp algorithm, another speedier version of gradient
  descent.

### C2W2L08
- Combining RMSProp and gradient descent with momentum: the Adam optimization
  algorithm (adaptive moment estimation).

### C2W2L09
- Utility of learning rate decay.
- Learning rate decay methods: automatic and manual decay methods.

## Week 3

### C2W3L01
- Overview of what is important to tune: learning rate, momentum term, number of
  hidden units, and mini-batch size, number of layers, and learning rate decay
  (all in the order of importance).
- Introduction to schemes regarding testing different combinations of
  hyperparamters (i.e. the hyperparamter search) -- using the vertices of a grid
  versus using random values (course-to-fine scheme), and prioritization of what
  to test more values of than others.

### C2W3L02
- Coarse-to-fine scheme: choice of scale (do not always want to sample *uniformly*
  at random).
- When it is appropriate to sample values uniformly at random. E.g. number of
  layers {2,3,4} -- there is no need for a special focus at any "end" of this set.
- When it is not. E.g. the learning rate, (0, 1] -- there is finer gradation
  near 0, therefore considering (uniform) random sampling on a log scale.
  Note that for hyperparameters relating to the exponentially-weighted average,
  we would like to more finely consider options near 1, instead of near 0.
- Related implementation.

### C2W3L03
- Tuning in practice -- training models in parallel vs babysitting, comparison
  of cost function trajectories.   

### C2W3L04
- Mention of normalizing inputs to speed up learning.
- About normalizing output volumes of any given layer to speed learning.
- Normalize before activation.
- Implementation of (mini-)batch normalization: using learnable parameters for
  normalization instead of necessarily the sample mean and variance
- The fact that bias is not needed if we use batch normalization.

### C2W3L05
- The problem of covariate shift in context of a neural network (namely, how
  learning of earlier layer w.r.t. the normalizing constants affects the input
  volume of the later layers), and the manner in which batch normalization
  alleviates it.
- The slight regularization effect caused by batch normalization due to the fact
  that the mean and variance of each batch is a slightly noisy estimate of that
  of the testing set, meaning that batch normalization, like dropout, adds some
  noise to the hidden layer's activations.
- Regularization effect not as strong as that caused by dropout, and decreases
  with batch size.

### C2W3L06
- On batch normalization at test time: when testing one example at a time, we
  cannot reasonably estimate mean and variance, so we can take an
  exponentially-weighted average across the testing mini-batches.

### C2W3L07
- Mechanics of the softmax regression function.

### C2W3L08
- Visualization of the decision boundaries of the softmax regression function,
  in the artificial example of a NN without hidden layers.

### C2W3L09
- Affirmation of the fact that Softmax with two layers is equivalent to logistic
  regression.
- Implementation of softmax, relation to backpropogation.

### C2W3L10
- Local optima are very uncommon in high-dimensional spaces.
- Plateaus (regions where many partials are close to 0) are the problem for
  gradient descent. 
- Adam &c. help us get off plateaus faster.

