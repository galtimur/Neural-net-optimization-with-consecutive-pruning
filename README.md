# Neural-net-optimization-with-consecutive-pruning
Neural net optimization with consecutive pruning

Pruning is the practice of removing parameters (which may entail removing individual parameters, or parameters in groups such as by neurons) from an existing network. Usually the goal of this process is to maintain accuracy of the network while increasing its efficiency.

This pet-project intends to explore whether the concequative decreasing of pruning ratio of the neural net (NN) can enhance performance of the model on the out-of-distribution (OOD) data and decrease training time. 

Usual pruning methods realize top-to-bottom approach. Firstly the full NN is trained, and then NN consequently pruned and traind to reach most sparse model (having least number of nonzero parameters) that still approximately retains the performance of the original model.

Here I try to apply upside down approach. I begin from training very sparse model and then increase the density of the model graduly, repeating the thaining. The hypothesis is that such procedure should lead to a better performance of the model on the OOD data and faster convergence in terms of FLOPS needed for training.

First part of the project is devoted to the analysis of the synthetic data produced by 3-layer perceptron. <br />
Description of the results and more user-friendly main section coming soon. <br />
At the future part I am going to test hypothesis on the image classification tasks using large CNN models. <br />
At the next step I am going to apply the approach to the transformer-like archetectures. <br />

### Files:

- Prunning main.py - Main programm
- mlp_setup.py - Сlasses for models of multilayer perceptron and functions for train data generation
- plotting_results.py - Аunctions for plotting results etc.
- prun_functions.py - Functions for pruning and repruning the models
