# Vector-Quantized-Naive-Bayes
Implementation of the vector-quantized naive Bayes (VQNB) model in Julia. 

Work from UBC's CPSC 540 course, Advanced Machine Learning. Note: minor parts of the code (such as the function to compute squared distances, the K-Means implementation, and the sample data file) are from the course, i.e. not created by me. I implemented the vector-quantized naive Bayes method (the VQNB.jl file). 

Ordinary naive Bayes is a probabilistic binary classification model. One of it's downfalls is that it does not account for variability *within* those classes. Take for example the mnist35.jld sample dataset in this repository. It is a collection of labelled, hand-written digits 3 and 5. But there are many different ways people might draw a 3 or a 5. VQNB addresses this by including a latent variable z, which is determined by K-Means clustering. The hope is that these clusterings will represent the "K different ways to draw a 3", and the "K different ways to draw a 5". 

Here is a plot of the probabilities of the features (pixels) P(x_j | y,z) using this particular MNIST example, using a value of k=5 for the clustering:

![alt text](https://github.com/justin-furlotte/Vector-Quantized-Naive-Bayes/blob/main/mnist35_example/mnist35_example.png)

On this particular dataset, VQNB scored a test error of 0.04 (far from the cutting edge on MNIST digits, but still useful to show that it does significantly better than ordinary naive Bayes, which got a test error of 0.10).

Unfortunately github does not currently support LaTeX rendering, so download the "READ-ME.pdf" for a mathematical description (with equations), or simply read a screenshot of it here:

![alt text](https://github.com/justin-furlotte/Vector-Quantized-Naive-Bayes/blob/main/vqnb_math.png)

