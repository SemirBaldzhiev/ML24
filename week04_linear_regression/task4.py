from task2 import Mat
import numpy as np

def mat_sig(mat: Mat) -> Mat:
    res = Mat(mat.rows, mat.cols)
    res.matrix = 1 / (1 + np.exp(-mat.matrix))
    return res

if __name__ == "__main__":
    w1 = Mat(2, 2)
    w2 = Mat(2, 2)

    w1.set_at(0, 0, 1)
    w1.set_at(0, 1, 2)
    w1.set_at(1, 0, 3)
    w1.set_at(1, 1, 4)

    w2.set_at(0, 0, 5)
    w2.set_at(0, 1, 6)
    w2.set_at(1, 0, 7)
    w2.set_at(1, 1, 8)
    
    print(f"Before: {w1}")
    print(f"After sigmoid: {mat_sig(w1)}")
    
    print(f"Before: {w2}")
    print(f"After sigmoid: {mat_sig(w2)}")