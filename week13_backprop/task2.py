class Value:
    def __init__(self, num: float):
        self.num = num

    def __repr__(self) -> str:
        return f"Value(data={self.num})"
    
    def __add__(self, other):
        return Value(self.num+other.num)



def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    result = x + y
    print(result)   
  
if __name__ == "__main__":  
    main()
