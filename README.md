# ANN---Writing-Number

该仓库记录自己手写的两个用于数字识别的前馈型神经网络，分别是C++版本和Python版本的.

## C++ Version

This repository reserves the ANN code, training data and testing data.

The training data are `train.in` and `train.out`(`train.zip` is `train.in`. It's so large~(102MB)). This training data contains 60000 examples. `train.in` has a matrix with the size of 60000\*784, each row is stretched by an image with 28\*28 pixels. 'train.out' has a vector with the size of 60000\*1, which are the answers corresond to each row of `train.in`.

The testing data are `test.in` and `test.out`, which contains 10000 examples and has the same data format as the training data, but they are new data comparing to the training data.

Finally, 2022.1.30. I found the biggest problem that is the initial value, when I setup the initial value of w, b in [-1, 1] randomly. The accuracy improves to 94% in 10 minutes, it's perfect, that means my code is right!!!

## Python Version

使用Python实现感知器算法和前馈神经神经网络

`Perceptron_main.py`为感知器算法.

`Input.py`为数据读入的部分代码, `timer.py`为用于计时的部分代码, `ANN.py`为前馈神经网络核心代码, `Main.py`为主程序. 使用主函数训练前请先**解压**训练数据及样本的压缩包`Data.rar`至与代码相同的目录下.

`Draw`文件夹中是对应报告中的训练图片生成的代码.

更详细的解释前馈神经网络参考报告[多层神经网络的训练问题.pdf](./多层神经网络的训练问题.pdf)

下面是核心算法实现原理图：

![前馈神经网络算法.png](https://s2.loli.net/2022/08/02/ctwBsjfE6LzOup8.png)

![传播算法流程图.png](https://s2.loli.net/2022/08/02/bXcCvnd8jq1JsMx.png)