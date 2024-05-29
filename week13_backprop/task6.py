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
    

def trace(val):
    nodes = set()
    edges = set()
    def _trace(val):
        if val in nodes:
            return
        nodes.add(val)
        for prev in val._prev:
            edges.add((prev, val))
            _trace(prev)
    _trace(val)
    return nodes, edges
    
        
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')
    
main()

# x
# nodes={Value(data=2.0)}
# edges=set()
# y
# nodes={Value(data=-3.0)}
# edges=set()
# z
# nodes={Value(data=10.0)}
# edges=set()
# result
# nodes={Value(data=10.0), Value(data=-3.0), Value(data=4.0), Value(data=-6.0), Value(data=2.0)}
# edges={(Value(data=-6.0), Value(data=4.0)), (Value(data=10.0), Value(data=4.0)), (Value(data=-3.0), Value(data=-6.0)), (Value(data=2.0), Value(data=-6.0))}
