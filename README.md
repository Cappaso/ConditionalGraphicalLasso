# ConditionalGraphicalLasso

Demo code of our CVPR 2016 paper "Conditional Graphical Lasso for Multi-label Image Classification"

%% Code authors: Qiang Li

%% Release time: May 4th, 2017

%% Current version: CGL_v1


1. Run the Main code.

Note the difference between 'script_CGL_CV.m' and 'script_CGL_TT.m',

the first corresponds to cross-validation based experiments,

the second corresponds to train/test based experiments.

The datasets and results are in 'data/' folder.

2. Datasets and Feature normalization.

We only used MULANscene in this demo. For other datasets, please follow the guidance in our CVPR paper.

For MULANscene and XXXX-PHOW, better to use "whitening" normalization.

For XXXX-CNN, better to apply no normalization.

3. Two methods are implemented.

cond_graph_lasso.m, corresponds to the algorithm using maximum pseudo-likelihood estimation.

CGL_Learning.m & CGL_Inference, correspond to the algorithms using mean-field variational inference.

%% Reference noticement:

If you have used the code, please kindly cite the following paper:

[1] Qiang Li, Maoying Qiao, Wei Bian, and Dacheng Tao,

"Conditional Graphical Lasso for Multi-label Image Classification",

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2016. (Poster)

%% Supporting information:

If any questions and comments, feel free to send your email to

Qiang Li (leetsiang.cloud@gmail.com)