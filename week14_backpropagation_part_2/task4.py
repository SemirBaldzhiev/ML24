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


def main() -> None:
        a = Value(2.0, label='x')
        b = Value(-3.0, label='y')
        c = Value(10.0, label='z')
        f = Value(-2.0, label='f')
        
        a.data += 0.01*6
        b.data += 0.01*(-4)
        c.data += 0.01*(-2)
        f.data += 0.01 * 4
        

        e = a * b
        e.label = 'e'
        
        e.data += 0.01*(-2)
        

        d = e + c
        d.label = 'd'
        
        d.data += 0.01 * (-2)
        
        L = d * f
        L.label = 'L'
        
        print(L.data)
        # draw_dot(L).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()
