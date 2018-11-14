
# Pytorch_Gans
Implementation Gans using Pytorch

## Introduction
To have a better understanding of gans, I am trying to implement some   
famous gans with Pytorch(0.4.1).
 In this process, I have referenced some others' githubs. Thank you very much!


## GANs

1. ### Standard Gan 
[Code](https://github.com/Septembit/Pytorch_Gans/blob/master/StandardGan.py)   [Paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

This is the most basic gans, which is implemented by only linear neural networks.

2. ### Deep Convolutional Gan(DCGAN)
[Code](https://github.com/Septembit/Pytorch_Gans/blob/master/DCGAN.py)   [Paper](https://arxiv.org/abs/1511.06434)

   #### Results
   Dataset is CIFAR10 and MINIST respectively.

(1) CIFAR10


![Have to say, it is really hard to generate perfect images](https://github.com/Septembit/Pytorch_Gans/blob/master/images/DCGAN_cifar.jpg)


![the loss of discriminator](https://github.com/Septembit/Pytorch_Gans/blob/master/images/DCGAN_cifar_Dloss.png)


![the loss of generator](https://github.com/Septembit/Pytorch_Gans/blob/master/images/DCGAN_cifar_Gloss.png)
