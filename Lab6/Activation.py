import numpy as np

class Activation():
    def __init__(self, activation_function, loss_function):
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.cache = None

    def forward(self, Z):
        if self.activation_function == "sigmoid":
            """
            Implements the sigmoid activation in numpy

            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation

            Returns:
            A -- output of sigmoid(z), same shape as Z
            """

            # GRADED FUNCTION: sigmoid_forward
            ### START CODE HERE ###
            def sigmoid(z):
                res = np.zeros(z.shape)
                pos_mask = z >= 0
                res[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
                
                neg_mask = ~pos_mask
                exp_term = np.exp(z[neg_mask])
                res[neg_mask] = exp_term / (1 + exp_term)
                return res
            A = sigmoid(Z)
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "relu":
            """
            Implement the RELU function in numpy
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            Returns:
            A -- output of relu(z), same shape as Z

            """

            # GRADED FUNCTION: relu_forward
            ### START CODE HERE ###
            def relu(z):
                return np.maximum(z, 0)
            A = relu(Z)
            self.cache = Z
            ### END CODE HERE ###

            assert(A.shape == Z.shape)

            return A
        elif self.activation_function == "softmax":
            """
            Implements the softmax activation in numpy

            Arguments:
            Z -- np.array with shape (n, C)
            self.cache -- stores Z as well, useful during backpropagation

            Returns:
            A -- output of softmax(z), same shape as Z
            """

            # GRADED FUNCTION: softmax_forward
            ### START CODE HERE ###
            def softmax(z):
                shifted_z = z - np.max(z, axis=1, keepdims=True)
                exp_scores = np.exp(shifted_z)
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            A = softmax(Z)
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "linear":
            """
            Linear activation (returns Z directly).
            """
            self.cache = Z.copy()
            return Z

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")


    def backward(self, dA=None, Y=None):
        if self.activation_function == "sigmoid":
            """
            Implement the backward propagation for a single SIGMOID unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the loss with respect to Z
            """

            # GRADED FUNCTION: sigmoid_backward
            ### START CODE HERE ###
            def sigmoid(z):
                res = np.zeros(z.shape)
                pos_mask = z >= 0
                res[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
                
                neg_mask = ~pos_mask
                exp_term = np.exp(z[neg_mask])
                res[neg_mask] = exp_term / (1 + exp_term)
                return res
            def sigmoid_derivative(z):
                s = sigmoid(z)
                res = s * (1 - s)
                return res
            Z = self.cache
            dZ = dA * sigmoid_derivative(Z)
            ### END CODE HERE ###

            assert (dZ.shape == Z.shape)

            return dZ

        elif self.activation_function == "relu":
            """
            Implement the backward propagation for a single RELU unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the loss with respect to Z
            """

            # GRADED FUNCTION: relu_backward
            ### START CODE HERE ###
            def relu_derivative(z):
                res = np.zeros(z.shape)
                mask = z > 0
                res[mask] = 1
                res[~mask] = 0
                return res
            Z = self.cache
            dZ = dA * relu_derivative(Z) + 0
            ### END CODE HERE ###

            assert (dZ.shape == Z.shape)

            return dZ

        elif self.activation_function == "softmax":
            """
            Implement the backward propagation for a [SOFTMAX->CCE LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """

            # GRADED FUNCTION: softmax_backward
            ### START CODE HERE ###
            def softmax(z):
                shifted_z = z - np.max(z, axis=1, keepdims=True)
                exp_scores = np.exp(shifted_z)
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            Z = self.cache
            s = softmax(Z)
            dZ = s - Y
            ### END CODE HERE ###

            assert (dZ.shape == self.cache.shape)

            return dZ

        elif self.activation_function == "linear":
            """
            Backward propagation for linear activation.
            """
            return dA

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")