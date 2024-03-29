# Deep-Reinforced-Tree-Traversal
This is the official release for paper **A Deep Reinforced Tree-traversal Agent for Coronary Artery Centerline Extraction**. 
Here we released detailed codes and also a set of toy models in order to visualize the result. Please check the original paper (https://doi.org/10.1007/978-3-030-87240-3_40) for detailed ideas.

![alt text](images/pipeline.png "pipeline")

## Requirements:
Please create an environment following requirements.txt to run the code.
One may need to change the pytorch version according to your own CUDA version.

## Usages:
Download the example_data from the link: https://drive.google.com/file/d/1yeJIoBALUGasHyFHAijkNILTtjhwfGXx/view?usp=sharing. Then substitute the place-holder folder with the one you downloaded.

1. To check the effect of the proposed method, run the inference through:
```shell
python tracer/inference.py
```
2. One can also run the train code with the toy data. However it's not likely to get any reasonbale result or weight:
```shell
python tracer/main.py
```
3. Train the discriminator with the following command. Still no sensable result is guaranteed:
```shell
python discriminator/main.py
```
## More Words:
For those who are truly interested in DRL, please reference https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch for more details. 
And honestly speaking, this code is a little bit messy and surely there are more elegent ways to organize the code as well as data structure. However, due to many reasons (mainly because I am too lazy	:ghost:	:ghost:	:ghost:) here we are. So try not to stuck in detailed codes. Feel free to contact me (zl502@cs.rutgers.edu) if you have any confusion.

## Reference
If you find this repository helpful, please consider giving a star and citing the following paper:
```
@InProceedings{10.1007/978-3-030-87240-3_40,
author="Li, Zhuowei and Xia, Qing and Hu, Zhiqiang and Wang, Wenji and Xu, Lijian and Zhang, Shaoting",
title="A Deep Reinforced Tree-Traversal Agent for Coronary Artery Centerline Extraction",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
year="2021",
}
```
