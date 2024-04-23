import numpy as np


class Linear:
    """
        The linear module in a neural network.

    """

    def __init__(self, input_dim, output_dim):
        self.params = dict()
        self.gradient = dict()
        # He initialization
        fan_in = input_dim
        std_dev = np.sqrt(2.0 / fan_in)
        self.params['W'] = np.random.randn(input_dim, output_dim) * std_dev
        self.params['b'] = np.zeros((1, output_dim))
        # self.params['W'] = np.random.normal(0, 0.1, size=(input_dim, output_dim))
        # self.params['b'] = np.random.normal(0, 0.1, size=(1, output_dim))

        self.gradient['W'] = np.zeros((input_dim, output_dim))
        self.gradient['b'] = np.zeros((1, output_dim))

    def forward(self, x):
        """
            The forward pass of the linear module.

        """

        forward_output = x @ self.params['W'] + self.params['b']
        return forward_output

    def backward(self, x, grad):
        """
            The backward pass of the linear module.

        """
        self.gradient['W'] = x.T @ grad
        self.gradient['b'] = np.sum(grad, axis=0)
        backward_output = grad @ self.params['W'].T

        return backward_output


# 2. ReLU Activation
class ReLU:
    """
        The ReLU module.

    """

    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray):
        """
            The forward pass of the ReLU module.

        """
        self.mask = x > 0
        forward_output = x * self.mask

        return forward_output

    def backward(self, x, grad):
        """
            The backward pass of the ReLU module.

        """
        backward_output = grad * self.mask
        return backward_output


class SoftmaxCrossEntropy:
    """ Softmax followed by cross-entropy loss.
    Generate a probability distribution over classes and calculate the loss.

    """
    def __init__(self):
        self.y_onehot = None
        self.logits = None
        self.sum_exp_logits = None
        self.prob = None

    def forward(self, x, y):
        self.y_onehot = np.zeros(x.shape).reshape(-1)
        self.y_onehot[y.astype(int).reshape(-1) - 1 + np.arange(x.shape[0]) * x.shape[1]] = 1.0
        self.y_onehot = self.y_onehot.reshape(x.shape)

        self.logits = x - np.max(x, axis=1, keepdims=True)  # normalize to prevent overflow
        self.sum_exp_logits = np.sum(np.exp(self.logits), axis=1, keepdims=True)
        self.prob = np.exp(self.logits) / self.sum_exp_logits

        forward_output = - np.sum(np.multiply(self.y_onehot, self.logits - np.log(self.sum_exp_logits))) / x.shape[0]
        return forward_output

    def backward(self, x, y):
        # derivative (p_i - y_i)
        backward_output = (self.prob - self.y_onehot) / x.shape[0]
        return backward_output
