# ANN---Writing-Number
This repository reserves the ANN code, training data and testing data.

The training data are `train.in` and `train.out`(`train.zip` is `train.in`. It's so large~(102MB)). This training data contains 60000 examples. `train.in` has a matrix with the size of 60000\*784, each row is stretched by an image with 28\*28 pixels. 'train.out' has a vector with the size of 60000\*1, which are the answers corresond to each row of `train.in`.

The testing data are `test.in` and `test.out`, which contains 10000 examples and has the same data format as the training data, but they are new data comparing to the training data.

Finally, 2022.1.30. I found the biggest problem that is the initial value, when I setup the initial value of w, b in [-1, 1] randomly. The accuracy improves to 94% in 10 minutes, it's perfect, that means my code is right!!!
