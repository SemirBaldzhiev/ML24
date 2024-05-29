
class Value:
    def __init__(self, num: float, prev=(), op=""):
        self.num = num
        self._prev=set(prev)
        self._op = op
        
    def __repr__(self) -> str:
        return f"Value(data={self.num})"
    
    def __add__(self, other):
        return Value(self.num+other.num, (self, other), op="+")
    
    def __mul__(self, other):
        return Value(self.num*other.num, (self, other), op="*")

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)

if __name__ == "__main__":
    main()