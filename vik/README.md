# Primis Dataset

This documents the trials on primis dataset using pytorch, version 0.3.

Here is a table of accuracies achieved thus far, and todo items:

## CPU: 100 epochs

- Learning rate 1e-5
- Batch sizes 256

This took running time of 2h 53 mins, and yield accuracy of **99.9**.

TODO List

- [ ] Use a learning rate with decay
- [ ] Run it on a GPU to assess speed, and to identify optimal batch size for
  max. utilisation.
- [ ] Visualise the accuracies, and saturation on matplotlib
- [ ] Visualise the CN layer outputs, and the ones with errors on the test set.
- [X] Practice with output on FH
