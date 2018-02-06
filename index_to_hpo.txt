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
- Gradient checking.  *Seems unnecessary if we are to use tensorflow, which
  implements backpropagation automatically when an optimization function is
specified.*
