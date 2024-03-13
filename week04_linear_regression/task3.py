from task2 import Mat
import numpy as np

def mat_mult(first_mat: Mat, second_mat: Mat) -> Mat:
    res = Mat(first_mat.rows, second_mat.cols)
    np.matmul(first_mat.matrix, second_mat.matrix, out=res.matrix)
    return res

def mat_add(first_mat: Mat, second_mat: Mat) -> Mat:
    res = Mat(first_mat.rows, first_mat.cols)
    np.add(first_mat.matrix, second_mat.matrix, out=res.matrix)
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

    print(f'Result of addition: {mat_add(w1, w2)}')
    print(f'Result of multiplication: {mat_mult(w1, w2)}')

