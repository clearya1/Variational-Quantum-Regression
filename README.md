# Variational Quantum Regression (VQR)

## Author  

Andrew Cleary, 2020 @ Irish Centre for High End Computing.
Supervised by Dr. Lee O'Riordan.

## Overview 

Creating a protocol for n-th order polynomial regression which can exploit the properties of a quantum computer. This is one of the simplest problems in machine learning. We make use of the principle of variational quantum computing, such that this protocal can be run on current NISQ hardware. 

## Calling Sequence

To enter the data sets, go into the code and set the x and y variables to their corresponding data sets. This process can of course be optimised or automated. Also, set the order variable to whatever order polynomial you wish to fit the data sets, with 'order = 1' meaning linear. Finally, one can also choose whichever classical optimiser they like, although we found that the 'BFGS' optimiser worked the best for this problem. Then, it is simply a matter of calling the following command through the terminal: 'python vfit_npoly.py'.

In summary, at the minimum, one just has to set the following three variables: x, y and order. 

## Jupyter Notebook

For a run through of the theory behind variational quantum computing, computing inner products with qubits and the inner-workings of the code, please refer to the jupyter notebook, 'Variational Quantum Regression.ipynb'

## Contents of the Repository

The main code is found in the 'vfit_npoly.py' Python script. The 'Testing Results' directory contains a cubic fit test to a dataset with random fluctuations. It also contains a map of the variation of the cost function, which is explained in the Jupyter notebook at length. In this map, we fit a linear line to y = 2*x, and plot the linear and constant co-efficient on the x- and y-axes respectively. The colour then corresponds to the value of the cost function. 

The 'Scaling Results' directory shows how the VQR protocol compares with the classical regression algorithm, 'polyfit' from NumPy. Time is plotted on the y-axis and the size of the datasets is plotted on the x-axis. The important thing to take away from this plot is that the quantum protocol scales linearly, just like the classical protocol. As we were running the VQR method through a simulator, we cannot truly compare the scaling coefficients with the classical protocol, as we would expect the VQR method to run much more quickly when actually run on quantum hardware!

