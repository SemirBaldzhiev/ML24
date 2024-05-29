class Value:
    def __init__(self, num: float):
        self.num = num

    def __repr__(self) -> str:
        return f"Value(data={self.num})"

    

def main() -> None:
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)
    
 
if __name__ == "__main__":
    main()