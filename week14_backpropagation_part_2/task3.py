import graphviz

class Value:
    def __init__(self, data: float, label="", prev=(), op=""):
        self.data = data
        self._prev=set(prev)
        self._op = op
        self.label = label
        self.grad = 0.0
        
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data+other.data, "", (self, other), op="+")
    
    def __mul__(self, other):
        return Value(self.data*other.data, "", (self, other), op="*")

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


def manual_der():
    h = 0.001
    
    a = Value(2.0, label='x')
    b = Value(-3.0, label='y')
    c = Value(10.0, label='z')
    f = Value(-2.0, label='f')
    e = a * b
    d = e + c
    old_l = d * f
    
    
    a = Value(2.0, label='x')
    b = Value(-3.0, label='y')
    c = Value(10.0, label='z')
    f = Value(-2.0, label='f')
    e = a * b
    d = e + c
    d.data += h
    new_l = d * f
    
    print((new_l.data - old_l.data) / h)
    

def main() -> None:
    manual_der()
        # a = Value(2.0, label='x')
        # b = Value(-3.0, label='y')
        # c = Value(10.0, label='z')
        # f = Value(-2.0, label='f')

        
        
        # e = a * b
        # e.label = 'e'

        # d = e + c
        # d.label = 'd'
        
        # L = d * f
        # L.label = 'L'
        
        # h = 0.001
        # a += h
        
        # e = a * b
        # e.label = 'e'

        # d = e + c
        # d.label = 'd'
        
        # L = d * f
        # L.label = 'L'
        
        
        # d1 = (L_a - L)/h

        # draw_dot(L).render(directory='./graphviz_output', view=True)

if __name__ == '__main__':
    main()

# manual backprop
# a = 2, b = -3, c = 10, f = -2
# L = d * f
# d = e + c
# e = a * b
# 
# dL/dd = f = -2
# dL/df = d = -6 + 10 = 4
# 
# dL/de = dL/dd * dd/de = -2 * 1 = -2
# dL/dc = dL/dd * dd/dc = -2
# dL/db = dL/de * de/db = -2 * 2 = -4
# dL/da = dL/de * de/da = -2 * (-3) = 6
