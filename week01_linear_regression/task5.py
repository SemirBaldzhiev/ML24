import numpy as np

# f(x1, x2) = w1*x1 + w2*x2

def create_or_dataset() -> list: 
    return [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]
    
def create_and_dataset() -> list:
    return [(0,0,0), (0,1,0), (1,0,0), (1,1,1)] 

def initialize_weights(x: int, y: int) -> float:
    n = np.random.random()*(y - x) + x
    return n

def calculate_loss(w1: float, w2: float, data: list) -> float:
    loss = 0
    for d in data:
        loss += (d[2] - (w1*d[0]+w2*d[1]))**2
    return loss/len(data)

def initialize_params() -> tuple:
    w1 = initialize_weights(0, 10)
    w2 = initialize_weights(0, 10)
    return w1, w2

def train_model(w1: float, w2: float, dataset: list) -> None:
    eps = 0.001
    learning_rate = 0.08
    
    for _ in range(1000):
        loss = calculate_loss(w1,w2, dataset)
        partial_derivative_w1 = (calculate_loss(w1 + eps, w2, dataset) - loss) / eps
        partial_derivative_w2 = (calculate_loss(w1, w2 + eps, dataset) - loss) / eps
        w1 -= learning_rate * partial_derivative_w1
        w2 -= learning_rate * partial_derivative_w2
        print(f'{w1=}, {w2=}, {loss=}')

def or_model() -> None:
    w1, w2 = initialize_params()
    or_dataset = create_or_dataset()
    train_model(w1, w2, or_dataset)
    

def and_model() -> None:
    w1, w2 = initialize_params()
    and_dataset = create_and_dataset()
    train_model(w1, w2, and_dataset)

def main() -> None:
    print("-------------------OR MODEL-------------------")
    or_model()
    print("-------------------AND MODEL-------------------")
    and_model()

if __name__ == "__main__":
    main()
    


