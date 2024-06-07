import math
import graphviz

class Value:
    def __init__(self, data: float, label="", prev=(), op=""):
        self.data = data
        self._prev=set(prev)
        self._op = op
        self.label = label
        self.grad = 0.0
        self._backward = None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        res = Value(self.data+other.data, "", (self, other), op="+")
        
        def add_backward():
            self.grad = res.grad
            other.grad = res.grad
        
        res._backward = add_backward
        
        return res 

    def __mul__(self, other):
        
        res = Value(self.data*other.data, "", (self, other), op="*")
        
        def mult_backward():
            self.grad = other.data * res.grad
            other.grad = self.data * res.grad
        
        res._backward = mult_backward
        
        return res
    
    def tanh(self):
        
        res = Value((math.exp(2*self.data) - 1) / ((math.exp(2*self.data) + 1)), "", (self, ), op="tanh")
        
        def tanh_backward():
            self.grad = 1 - res.data**2
        
        res._backward = tanh_backward
        return res
    
    def backward(self):
        pass
        
                     

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

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    print(nodes)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ {n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}}}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def top_sort(start: Value) -> list[Value]:
    result = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            result.append(v)
    build(start)
    return result


def main() -> None:
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    
    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    
    x2w2 = x2 * w2
    x2w2.label = "x1w1"
    
    z = x1w1 + x2w2
    z.label = "x1w1 + x2w2"
    
    logit = z + b
    logit.label = "logit"
    
    L = logit.tanh()
    L.label = "L"
    L.grad = 1.0
    
    logit.grad = 1 - logit.tanh().data**2
    
    b.grad = logit.grad
    z.grad = logit.grad
    
    x1w1.grad = z.grad
    x2w2.grad = z.grad
    
    x1.grad = w1.data*x1w1.grad
    w1.grad = x1.data*x1w1.grad
    
    x2.grad = w2.data*x2w2.grad
    w2.grad = x2.data*x2w2.grad
        
    draw_dot(L).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()