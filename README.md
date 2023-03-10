# Neural-net-optimization-with-consecutive-pruning
Neural net optimization with consecutive pruning

Pruning is the practice of removing parameters (which may entail removing individual parameters, or parameters in groups such as by neurons) from an existing network. Usually, the goal of this process is to maintain the accuracy of the network while increasing its efficiency.

This pet project intends to explore whether the consecutive decreasing of the pruning ratio of the neural net (NN) can enhance the performance of the model on the out-of-distribution (OOD) data and decrease training time. 

Usual pruning methods realize a top-to-bottom approach. Firstly the full NN is trained, and then NN is consequently pruned and trained to reach the most sparse model (having the least number of nonzero parameters) that still approximately retains the performance of the original model.

Here I try to apply an upside-down approach. I begin with training a very sparse model and then increase the density of the model gradually, repeating the training. The hypothesis is that such a procedure should lead to a better performance of the model on the OOD data and faster convergence in terms of FLOPS needed for training.

The first part of the project is devoted to the analysis of the synthetic data produced by a 3-layer perceptron. <br />
Description of the results and a more user-friendly main section coming soon. <br />
In the future part, I am going to test the hypothesis on the image classification tasks using large CNN models. <br />
In the next step, I am going to apply the approach to transformer-like architectures. <br />

### Files:

- Pruning_main.py - Main program
- mlp_setup.py - Сlasses for models of multilayer perceptron and functions for training data generation
- plotting_results.py - Аunctions for plotting results etc.
- prun_functions.py - Functions for pruning and repruning the models
