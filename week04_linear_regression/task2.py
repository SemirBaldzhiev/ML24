import numpy as np

class Mat:
    def __init__(self, rows: int, cols: int) -> None:
        self.__rows = rows
        self.__cols = cols
        self.__matrix = np.zeros((rows, cols))
        
    def __are_valid_idxs(self, r_idx: int, c_idx) -> bool:
        return (r_idx >= 0 and r_idx < self.__rows) and (c_idx >= 0 and c_idx < self.__cols)
        
    def set_at(self, row_idx: int, col_idx: int, value: float) -> None:
        if self.__are_valid_idxs(row_idx, col_idx):
            self.__matrix[row_idx, col_idx] = value
        else:
            print(f"Invalid position ({row_idx}, {col_idx})!")
        
    def randomize(self, start: float, end: float) -> None:
        self.__matrix = np.random.uniform(low=start, high=end,size=(self.__rows, self.__cols))
    
    @property
    def rows(self) -> int:
        return self.__rows
    
    @property
    def cols(self) -> int:
        return self.__cols
    
    @property
    def matrix(self) -> np.ndarray:
        return self.__matrix
    
    @matrix.setter
    def matrix(self, new_matrix) -> None:
        self.__matrix = new_matrix
    
    def __repr__(self) -> str:
        return f"{self.__matrix}"


if __name__ == "__main__":
    np.random.seed(42)

    inp = Mat(1, 2)
    w1 = Mat(2, 2)
    b1 = Mat(1, 2)
    w2 = Mat(2, 1)
    b2 = Mat(1, 1)

    inp.set_at(0, 0, 0)
    inp.set_at(0, 1, 1)

    w1.randomize(0, 1)
    b1.randomize(0, 1)
    w2.randomize(0, 1)
    b2.randomize(0, 1)

    print(inp)
    print(w1)
    print(b1)
    print(w2)
    print(b2)
