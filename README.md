# Uncertainty-Aware-Pytorch-Implem
The pytorch implementation of paper ["Uncertainty-Aware Attention for Reliable Interpretation and Prediction"](https://arxiv.org/abs/1805.09653) by Jay Heo, Hae Beom Lee, Saehoon Kim, Juho Lee. 
Based on its Tensorflow implementation from the original authors [here](https://github.com/jayheo/UA.git)

Requirements Python 3.7.x, Pytorch 1.1.x

Provided in dir dataset: Physionet in numpy format predicting mortatility risk
Other dataset can be found [here](https://physionet.org/physiobank/database/challenge/2012/) for Physionet for [MIMIC-III dataset](https://mimic.physionet.org/).


This implementation was able to achieve ~76% AUCROC on the provided evaluation dataset which is short of the ~77% claimed in the paper.
The configuration follows exactly the configuration in the original [implementation](https://github.com/jayheo/UA.git) by the authors.



