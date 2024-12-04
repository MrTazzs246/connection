import numpy as np

class MLP:
    def __init__(self, no_inputs : int, no_hidden_units : int, no_outputs : int) -> None:
        
        # node set up
        self.no_inputs = no_inputs
        self.no_hidden_units = no_hidden_units
        self.no_outputs = no_outputs 

        # biases & weights 
        self.W1 = np.zeros((no_inputs, no_hidden_units))                # input - hidden
        self.B1 = np.zeros(no_hidden_units)
        self.W2 = np.zeros((no_hidden_units, no_outputs))               # hidden - output 
        self.B2 = np.zeros(no_outputs)

        # deltas  
        self.dW1 = np.zeros((no_inputs, no_hidden_units))               # ΔW1 (input - hidden)
        self.dB1 = np.zeros(no_hidden_units)   
        self.dW2 = np.zeros((no_hidden_units, no_outputs))              # ΔW2 (hidden - output)
        self.dB2 = np.zeros(no_outputs)

        # activations
        self.Z1 = np.zeros(no_hidden_units)                             # hidden 
        self.Z2 = np.zeros(no_outputs)                                  # output 

        # neuron values 
        self.H = np.zeros(no_hidden_units)                              # hidden 
        self.O = np.zeros(no_outputs)                                   # output

    def randomise(self, scale : float) -> None:

        # intialise weights with random values
        self.W1 = np.random.uniform(scale * -1, scale, (self.no_inputs, self.no_hidden_units))
        self.B1 = np.random.uniform(scale * -1, scale,  self.no_hidden_units)
        self.W2 = np.random.uniform(scale * -1, scale, (self.no_hidden_units, self.no_outputs))
        self.B2 = np.random.uniform(scale * -1, scale,  self.no_outputs)
        
        # set deltas back to zero 
        self.dW1 = np.zeros((self.no_inputs, self.no_hidden_units)) 
        self.dB1 = np.zeros(self.no_hidden_units)
        self.dW2 = np.zeros((self.no_hidden_units, self.no_outputs))  
        self.dB2 = np.zeros(self.no_outputs)

    ## Non linear activation function for hidden layer
    def relu(self, x : np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        # determine positive vector 
        greater_than_zero = x > 0
        derivative = greater_than_zero.astype(float)
        return derivative
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

    ## Bounded [0, 1] activation function for output layer
    def sigmoid(self, x : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def linear(self, x: np.ndarray) -> np.ndarray:
        return x

    def linear_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def forward(self, input : np.ndarray) -> np.ndarray: 

        # hidden layer
        self.Z1 = np.dot(input, self.W1) + self.B1         # input * weights + bias --> hidden
        self.H = self.tanh(self.Z1)                     # activation? hidden layer

        # output layer
        self.Z2 = np.dot(self.H, self.W2) + self.B2     # (hidden layer output) * weights + bias --> output
        self.O = self.sigmoid(self.Z2)                  #  output layer (sigmoid) 
        return self.O

    def backward(self, input, targets : np.ndarray) -> float: 

        # distance between output and actual
        error = np.mean(np.square(targets - self.O))
        
        # deltas output layer
        delta_output = (self.O - targets) * (self.O * (1 - self.O))
        dW2_update = np.dot(self.H.T, delta_output)     # delta W2
        self.dW2 += dW2_update                          # sum delta W2
        self.dB2 = np.sum(delta_output, axis=0)

        # deltas hidden layer
        delta_hidden = np.dot(delta_output, self.W2.T) * self.tanh_derivative(self.H)          # Backpropagate deltas through W2
        dW1_update = np.dot(input.T, delta_hidden)                                          # Weight updates for W1
        self.dW1 += dW1_update                                                              # Accumulate weight updates for W1
        self.dB1 = np.sum(delta_hidden, axis=0)
        
        return error

    def update_weights(self, learning_rate : float) -> None: 
        
        # new weight = step * delta 
        self.W1 -= learning_rate * self.dW1
        self.B1 -= learning_rate * self.dB1
        self.W2 -= learning_rate * self.dW2
        self.B2 -= learning_rate * self.dB2

        # Reset deltas to zero
        self.dW1 = np.zeros_like(self.dW1)
        self.dW2 = np.zeros_like(self.dW2)
        self.dB1 = np.zeros_like(self.dB1)
        self.dB2 = np.zeros_like(self.dB2)


def train(NN : MLP, X : np.array, y : np.array) -> None:
    for t in range(1000): 
        output = NN.forward(X)
        error = NN.backward(X, y)
        if (t % 2) == 0: 
            NN.update_weights(0.01)
        print(f"Error at epoch {t} is {error}")

NN = MLP(2, 4, 1)

# training data 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
y = np.array([[0], [1], [1], [0]])   

NN.randomise(0.01)
train(NN, X, y)