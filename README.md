# Deep-Learning

Content: https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann

Deep Learning is a subset of Machine Learning, essentially a neural network with three or more hidden layers.
These neural networks mimic the behavior of the human brain, allowing it to learn from large amounts of data.

Through the process of gradient descent and back propagation, the model adjusts and fits itself for precise accuracy.

### Types of Neural Networks:
Convolutional NN’s:
	Used primarily in Computer Vision and image classification applications.
	Can detect features and patterns within an image like object detection.
Recurrent NN’s: 
	Are typically used in Natural Language processing and speech recognition applications as it leverages sequential and time-series data.

### Neuron:
  Basic building block of Artificial Neural Networks.
	The neuron has multiple inputs which may be either an input or a output from preceding neuron connected through synapse. These synapse carry weights. 
	In hidden layer neurons, The inputs from the preceding layers are multiplied by their corresponding weights and they are all added up. Then an activation function is applied on the resulting sum.       This value is passed onto either as output value or as an input to the next neuron. These weights are randomly initialized, but later are tuned during training process using Gradient Descent and Back Propagation techniques.  

<img width="1400" height="700" alt="image" src="https://github.com/user-attachments/assets/77186b20-12c7-4808-96a0-84a24010f885" />


### Activation Functions
Without activation functions, the entire neural network would behave like a linear model, regardless of its depth. Linear combinations of linear functions result in another linear function. Activation functions introduce non-linearity into the network, enabling it to learn and approximate complex, non-linear relationships in data.
Threshold Function:
Range - 0 or 1

<img width="975" height="472" alt="image" src="https://github.com/user-attachments/assets/b52b5363-458b-46b4-b63c-ff69f776903f" />


### Sigmoid Function: Range: [0,1], gives the probability of X = 1 i.e., P( X = 1 )

<img width="1200" height="630" alt="image" src="https://github.com/user-attachments/assets/ca83a7fb-2267-4557-9660-5b218c3fba7b" />

### Rectifier Function (ReLU): Range: [0,INF)

<img width="850" height="409" alt="image" src="https://github.com/user-attachments/assets/434085e4-48be-43fd-8d44-7fb02d6c4bcb" />

### Hyperbolic tangent Activation Function: Range: [-1 , 1]

<img width="846" height="449" alt="image" src="https://github.com/user-attachments/assets/fd9d8b4b-d166-4d9a-b618-a5f1f11ed988" />

### Linear or identity Activation function: Range: ( -INF, INF)

<img width="867" height="564" alt="image" src="https://github.com/user-attachments/assets/cea0070a-987f-4324-acfb-2089fad3819b" />

### Leaky ReLU:

<img width="690" height="488" alt="image" src="https://github.com/user-attachments/assets/9fad0d31-1bd1-4750-be31-12a302e163c8" />

### Softmax Activation Function:
	Used for Multi-class classification, where as in binary classification we can use sigmoid function.

Without activation functions, the entire neural network would behave like a linear model, regardless of its depth. Linear combinations of linear functions result in another linear function. Activation functions introduce non-linearity into the network, enabling it to learn and approximate complex, non-linear relationships in data.
Activation functions are crucial for the training process of neural networks, specifically in the back propagation algorithm. Non-linear activation functions introduce gradients, allowing the optimization algorithm (e.g., gradient descent) to adjust the weights during training. This enables the network to learn and adapt to the input data by minimizing the error or loss function.
Activation functions help in normalizing the output of neurons, preventing them from growing too large or too small. This can stabilize the learning process and prevent issues like vanishing gradients or exploding gradients, which can hinder the convergence of the training algorithm.

### How do Neural Networks Work?


### Cost Function
	It is a measure of “how good” a neural network did with respect to the given training sample and the expected output.

Cost Function can be represented as <img width="222" height="50" alt="image" src="https://github.com/user-attachments/assets/d9740ec6-3817-47d3-b4cd-1145bfad9036" />


	W - NN’s Weights 
	B - NN’s Biases
	S - input of single training sample
	E - desired output of that training sample

In back propagation, the error of the output layer is computed using the cost function, via

<img width="240" height="100" alt="image" src="https://github.com/user-attachments/assets/91f14108-a2f5-4b86-97fe-035442a08af7" />

### Cost Function Requirements:
To be used in back propagation, 
1. The Cost function C must be able to be written as an average over cost functions Cx for individual training samples, x. 

<img width="224" height="104" alt="image" src="https://github.com/user-attachments/assets/cff61778-4057-4979-b562-0c6535ecad25" />

It states that, the cost function is an average of the differences b/w the predicted/ model output and the actual value for all training samples.
2. The cost function must not be dependent on any activation values of the neural network besides the output values. This states that, the cost function is a function of model outputs and the actual values, therefore, it only depends on the model output and not on any intermediate values generated in the hidden layers.
					      
                C = C ( y , y^ )

### Convolutional Neural Networks

Gradient-based learning applied to document recognition: https://ieeexplore.ieee.org/document/726791
Introduction to Convolutional Neural Networks: https://cs.nju.edu.cn/wujx/paper/CNN.pdf
http://cs.nju.edu.cn/wujx/paper/CNN/pdf

### Convolution:
We have a filter/ kernel/ feature detector or size N x N ( N can be 3 or 5 or 7 it depends)
By using this kernel, we do the below operation.

<img width="1283" height="617" alt="image" src="https://github.com/user-attachments/assets/1cda67a4-e1ef-4a51-bdb6-89801bd1d138" />

The steps at which the kernel is moved is called stride.
The image on the right is called Feature Map/ activation map/ convolved feature.
The primary purpose of convolution is to find features in your image using the feature detector/ kernel.

### ReLU Layer:
To increase non linearity we use relu layer.

Understanding Convolutional Neural Networks with a Mathematical Model by C.C Jay Kuo:  https://www.sciencedirect.com/science/article/abs/pii/S1047320316302267

### Max Pooling:
Introduces spacial invariances.
Prevents over-fitting


<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/48ba3462-289f-4cbb-999d-5e0a80d57125" />


Evaluation of pooling operations in Convolutional Architecture for object recognition by Domino Scherer : https://link.springer.com/chapter/10.1007/978-3-642-15825-4_10


<img width="1087" height="355" alt="image" src="https://github.com/user-attachments/assets/c2ae744c-26da-4ce4-8a9e-6e1e0371e141" />


### Flatten:
Convert each pooled layer into one dimensional array to further feed it to the artificial Neural network.

<img width="1665" height="724" alt="image" src="https://github.com/user-attachments/assets/c1e96b55-54d2-4531-8110-2317ad04df73" />

### Fully Connected Layer:
The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3): https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

<img width="2500" height="846" alt="image" src="https://github.com/user-attachments/assets/ef66908a-455d-40e2-9d1e-565990b2284b" />

### SoftMax & Cross Entropy:
Softmax is used to convert output values of output neurons into probabilities.
It is often used with cross entropy loss function, which calculates loss in each iteration, and helps back propagation to adjust weights and feature maps. Cross entropy is preferred than MSE or basic classification error metrics as it would leverage the log function to scale up the minute errors during the initial iterations of back propagation, and helps to attain stability swiftly.


Further References: 
1. List of Cost functions used in neural networks alongside applications:
https://github.com/Bladefidz/machine-learning/blob/master/literatures/neural%20networks/a-list-of-cost-functions-used-in-neural-networks-alongside-applications.pdf

2. Gradient Descent:
https://iamtrask.github.io/2015/07/12/basic-python-network/

3. A friendly introduction to cross entropy loss, by Rob DiPietro:
https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/

5. How to implement a neural network Intermezzo 2, by Peter Roelants: 
https://peterroelants.github.io/posts/neural-network-implementation-part01/
https://peterroelants.github.io/posts/cross-entropy-softmax/

6. Ultimate Guide for CNN:
https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn

 
