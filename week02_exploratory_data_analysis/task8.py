import numpy as np

class Xor:
    def __init__(self):
        self.w1 = np.random.uniform(0, 3, size=(2, 2))
        self.b1 = np.random.uniform(0, 3, size=(1, 2))
        self.w2 = np.random.uniform(0, 3, size=(2, 1))
        self.b2 = np.random.uniform(0, 3, size=(1, 1))
        self.epochs = 100_000
        self.lr = 0.1
        self.expected_output = np.array([[0],[1],[1],[0]])

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs) -> np.array:
        for _ in range(self.epochs):
            hidden_layer = np.dot(inputs, self.w1) + self.b1
            hidden_layer_output = self.__sigmoid(hidden_layer)

            output_layer = np.dot(hidden_layer_output, self.w2) + self.b2
            predicted_output = self.__sigmoid(output_layer)

            loss = self.expected_output - predicted_output
            d_pred_out = loss * self.__sigmoid_derivative(predicted_output)

            loss_hidden_layer = d_pred_out.dot(self.w2.T)
            d_hidden_layer = loss_hidden_layer * self.__sigmoid_derivative(hidden_layer_output)
            
            self.w2 += hidden_layer_output.T.dot(d_pred_out) * self.lr
            self.b2 += np.sum(d_pred_out,axis=0,keepdims=True) * self.lr
            self.w1 += inputs.T.dot(d_hidden_layer) * self.lr
            self.b1 += np.sum(d_hidden_layer,axis=0,keepdims=True) * self.lr

        return predicted_output

def main():
    xor = Xor()
    output = xor.forward(np.array([[0,0],[0,1],[1,0],[1,1]]))
    print(output)

if __name__ == "__main__":
    main()
