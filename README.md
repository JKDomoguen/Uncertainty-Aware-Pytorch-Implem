# Uncertainty-Aware-Pytorch-Implem
The pytorch implementation of paper "Uncertainty-Aware Attention for Reliable Interpretation and Prediction" 
by Jay Heo, Hae Beom Lee, Saehoon Kim, Juho Lee. 
Based on its Tensorflow implementation from the original authors [here](https://github.com/jayheo/UA.git)

Requirements Python 3.7.x, Pytorch 1.1.x

Provided in dir dataset: Physionet in numpy format predicting mortatility risk
Other dataset can be found [here](https://physionet.org/physiobank/database/challenge/2012/) for Physionet for [MIMIC-III dataset](https://mimic.physionet.org/).



Note: Was not able to achieve the ~78% AUCROC on evaluation dataset claimed in the [paper](https://arxiv.org/abs/1805.09653). At most only achieved 72% AUC-ROC
