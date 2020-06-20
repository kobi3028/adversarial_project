# adversarial_project

In this work, we reproduce an adversarial detection method presented in the paper “Detecting Adversarial Perturbations Through Spatial Behavior in Activation Spaces” by Ziv Katzir and Yuval Elovici.

We managed to reproduce the paper configuration PCA+KNN in the python file code/detector_training.py 

In addition implemented our own configurations in the jupyter notebook code/autoencoders_and_anns.ipynb
Configurations: AutoEncoder with Sigmoid activation+ANN and AutoEncoder with Tanh activation+ANN.

From evaluating all of the configurations, the Tanh AutoEncoder had the best results.
The results graphs for each configuration are inside /data/#configuration_name directory