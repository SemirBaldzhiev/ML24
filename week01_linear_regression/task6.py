import numpy as np

# f(x1, x2) = w1*x1 + w2*x2 + b
# Diferrence between task5 and task6 is that in task6 we have bias term b 
# which helps us to shift the decision boundary and loss function converges faster.


def create_or_dataset() -> list: 
    return [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]
    
def create_and_dataset() -> list:
    return [(0,0,0), (0,1,0), (1,0,0), (1,1,1)] 

def initialize_weights(x: int, y: int) -> float:
    n = np.random.random()*(y - x) + x
    return n

def initialize_params() -> tuple:
    w1 = initialize_weights(0, 10)
    w2 = initialize_weights(0, 10)
    b = initialize_weights(0, 10)
    return w1, w2, b

def calculate_loss(w1: float, w2: float, b: float, data: list) -> float:
    loss = 0
    for d in data:
        loss += (d[2] - (w1*d[0]+w2*d[1]+b))**2
    return loss/len(data)

def train_model(w1: float, w2: float, b: float, dataset: list) -> None:
    eps = 0.001
    learning_rate = 0.08
    
    for _ in range(1000):
        loss = calculate_loss(w1, w2, b, dataset)
        partial_derivative_w1 = (calculate_loss(w1 + eps, w2, b,  dataset) - loss) / eps
        partial_derivative_w2 = (calculate_loss(w1, w2 + eps, b, dataset) - loss) / eps
        partial_derivative_b = (calculate_loss(w1, w2, b + eps, dataset) - loss) / eps
        w1 -= learning_rate * partial_derivative_w1
        w2 -= learning_rate * partial_derivative_w2
        b -= learning_rate * partial_derivative_b
        print(f'{w1=}, {w2=}, {loss=}')

def or_model() -> None:
    w1, w2, b = initialize_params()
    or_dataset = create_or_dataset()
    train_model(w1, w2, b, or_dataset)
    

def and_model() -> None:
    w1, w2, b = initialize_params()
    and_dataset = create_and_dataset()
    train_model(w1, w2, b, and_dataset)

def main() -> None:
    print("-------------------OR MODEL-------------------")
    or_model()
    print("-------------------AND MODEL-------------------")
    and_model()

if __name__ == "__main__":
    main()
    


