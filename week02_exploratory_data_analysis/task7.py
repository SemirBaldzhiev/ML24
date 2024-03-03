import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def create_dataset_and() -> list[(int, int, int)]:
    return [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]


def create_dataset_or() -> list[(int, int, int)]:
    return [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


def create_dataset_nand() -> list[(int, int, int)]:
    return [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

def initialize_weights(x: int, y: int) -> float:
    return np.random.uniform(x, y)

def calculate_loss(w1: int, w2: int, bias: int,
                   dataset: list[(int, int, int)]) -> float:
    sum = 0
    n = len(dataset)
    for (x, y, expected) in dataset:
        actual = x * w1 + y * w2 + bias
        sum += (actual - expected)**2
    return sum / n

def calculate_loss_sig(w1: int, w2: int, bias: int,
                   dataset: list[(int, int, int)]) -> float:
    sum = 0
    n = len(dataset)
    for (x, y, expected) in dataset:
        actual = sigmoid(x * w1 + y * w2 + bias)
        sum += (actual - expected)**2
    return sum / n

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

def plot_sigmoid() -> None:
    x = np.linspace(-10, 10, 50)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()

def plot_loss(loss: list[float]) -> None:
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss over epochs')
    plt.ylim(0, 0.01)
    plt.show()

def models_with_sigmoid() -> None:
    and_dataset = create_dataset_and()
    w1 = initialize_weights(0, 10)
    w2 = initialize_weights(0, 10)
    bias = initialize_weights(0, 10)
    
    loss_data = []

    eps = 0.1
    learning_rate = 0.1
    epochs = 100_000

    for i in range(epochs):
        loss = calculate_loss_sig(w1, w2, bias, and_dataset)
        der_loss_w1 = (calculate_loss_sig(w1 + eps, w2, bias, and_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss_sig(w1, w2 + eps, bias, and_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss_sig(w1, w2, bias + eps, and_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias
        loss_data.append(loss)
    
    plot_loss(loss_data)

    print(
        f'loss now is: {calculate_loss_sig(w1, w2, bias, and_dataset)} for and prediction'
    )
    print('and predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j,sigmoid(w1 * i + w2 * j + bias))

    or_dataset = create_dataset_or()
    w1 = initialize_weights(0, 3)
    w2 = initialize_weights(0, 3)
    bias = initialize_weights(0, 3)
    
    loss_data = []

    for i in range(epochs):
        loss = calculate_loss_sig(w1, w2, bias, or_dataset)
        der_loss_w1 = (calculate_loss_sig(w1 + eps, w2, bias, or_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss_sig(w1, w2 + eps, bias, or_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss_sig(w1, w2, bias + eps, or_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias
        loss_data.append(loss)

    plot_loss(loss_data)

    print(
        f'loss now is: {calculate_loss_sig(w1, w2, bias, or_dataset)} for or prediction'
    )
    print('or predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j, sigmoid(w1 * i + w2 * j + bias))
            
    
    nand_dataset = create_dataset_nand()
    w1 = initialize_weights(0, 3)
    w2 = initialize_weights(0, 3)
    bias = initialize_weights(0, 3)
    
    loss_data = []
    
    for i in range(epochs):
        loss = calculate_loss_sig(w1, w2, bias, nand_dataset)
        der_loss_w1 = (calculate_loss_sig(w1 + eps, w2, bias, nand_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss_sig(w1, w2 + eps, bias, nand_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss_sig(w1, w2, bias + eps, nand_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias
        loss_data.append(loss)
    
    plot_loss(loss_data)
    
    print(
        f'loss now is: {calculate_loss_sig(w1, w2, bias, nand_dataset)} for nand prediction'
    )
    print('nand predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j, sigmoid(w1 * i + w2 * j + bias))

def main() -> None:
    and_dataset = create_dataset_and()
    w1 = initialize_weights(0, 10)
    w2 = initialize_weights(0, 10)
    bias = initialize_weights(0, 10)

    eps = 0.1
    learning_rate = 0.1
    epochs = 100_000

    for _ in range(epochs):
        loss = calculate_loss(w1, w2, bias, and_dataset)
        der_loss_w1 = (calculate_loss(w1 + eps, w2, bias, and_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss(w1, w2 + eps, bias, and_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss(w1, w2, bias + eps, and_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias

    print(
        f'loss now is: {calculate_loss(w1, w2, bias, and_dataset)} for and prediction'
    )
    print('and predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j, w1 * i + w2 * j + bias)

    or_dataset = create_dataset_or()
    w1 = initialize_weights(0, 3)
    w2 = initialize_weights(0, 3)
    bias = initialize_weights(0, 3)

    for _ in range(epochs):
        loss = calculate_loss(w1, w2, bias, or_dataset)
        der_loss_w1 = (calculate_loss(w1 + eps, w2, bias, or_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss(w1, w2 + eps, bias, or_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss(w1, w2, bias + eps, or_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias

    print(
        f'loss now is: {calculate_loss(w1, w2, bias, or_dataset)} for or prediction'
    )
    print('or predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j, w1 * i + w2 * j + bias)
    
    print("="*50)
    # Models with sigmoid function
    models_with_sigmoid()
    
    # for i in range(-10, 11):
    #     print(f'{i} => {sigmoid(i)}')
    
    plot_sigmoid()

if __name__ == '__main__':
    main()

    
    