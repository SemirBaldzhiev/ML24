import numpy as np

np.random.seed(42)

def create_dataset(n):
    input = [x for x in range(n+1)]
    output = [2*x for x in range(n+1)]
    dataset = list(zip(input, output))
    return dataset
    # return [(x, 2*x) for x in range(n+1)]

def initialize_weights(x, y):
    n = np.random.random()*(y - x) + x
    return n

def calculate_loss(w, data):
    loss = 0
    for d in data:
        loss += (w*d[0] - d[1])**2
    return loss/len(data)

if __name__ == "__main__":
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))
    print(create_dataset(4))
    dataset = create_dataset(4)
    w = initialize_weights(0, 10)
    print(f'{w=}')
    print(calculate_loss(w + 0.001*2, dataset))
    print(calculate_loss(w + 0.001, dataset))
    print(calculate_loss(w, dataset))
    print(calculate_loss(w - 0.001, dataset))
    print(calculate_loss(w - 0.001*2, dataset))
    
    eps = 0.001
    learning_rate = 0.001
    
    for i in range(10000):
        approximate_derivative = (calculate_loss(w + eps, dataset) - calculate_loss(w, dataset)) / eps
        loss = calculate_loss(w, dataset)
        print(f'{w=}')
        print(f'{loss=}')
        w -= learning_rate * approximate_derivative
        loss = calculate_loss(w, dataset)
        print(f'{w=}')
        print(f'{loss=}')