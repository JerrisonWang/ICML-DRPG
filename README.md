# ICML - DRPG paper

This repository contains files that desribe the robust Markov decision processes used in the numerical example of the paper "Policy Gradient in Robust MDPs with Global Convergence Guarantee".

There are two folders . The folder `Garnet Problem` includes:

1. C++ codes for generating Garnet problems with different size,
2. C++ codes for soving the robust MDPs with our method DRPG and the benchmark method robust value iteration and,
3. Python codes for plotting the error decreasing performence.

The folder `Inventory Management Problem` includes:

1. Python codes for a simple function collections,
2. Python codes for generating a inventory problem with parameterized transition and applying DRPG to solve it and,
3. Two CSV files which specify the cost and the empirical transition of the inventory management problem.


