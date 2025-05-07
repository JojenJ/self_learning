import numpy as np
class node():
    def __init__(self,in_nodes=[]):
        self.in_nodes = in_nodes

        self.out_nodes = []

        self.value = None
        self.gradients = {}
        for n in self.in_nodes:
            n.out_nodes.append(self)

    def forward(self,value =None):
        if value !=None:
            self.value = value
    def backward(self):
        raise NotImplementedError
    

class add(node):
    def __init__(self, x=[]):
        node.__init__(self,x)
        self.num = len(x)

    def forward(self):
        sum =0
        for n in range(self.num):
            sum += self.in_nodes[n].value
        
        self.value = sum
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            grad_cost = n.gradients[self]
            for i in self.in_nodes:
                self.gradients[i] += grad_cost * 1  # ∂(x+y)/∂x = 1


class multi(node):
    def __init__(self, x=[]):
        node.__init__(self,x)
        self.num = len(x)
    def forward(self):
        result = 1
        for n in range(self.num):
            result = result*self.in_nodes[n].value
        self.value = result
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            grad_cost = n.gradients[self]
            for i, node in enumerate(self.in_nodes):
                # 所有输入的乘积除以当前输入值，即为当前输入的偏导
                prod = 1
                for j, other in enumerate(self.in_nodes):
                    if i != j:
                        prod *= other.value
                self.gradients[node] += grad_cost * prod

class Input(node):
    def __init__(self):
        # an Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        node.__init__(self)

    # NOTE: Input node is the only node that may
    # receive its value as an argument to forward().
    #
    # All other node implementations should calculate their
    # values from the value of previous nodes, using
    # self.in_nodes
    #
    # Example:
    # val0 = self.in_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.out_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Linear(node):
    def __init__(self, X, W, b):
        node.__init__(self, [X, W, b])


    def forward(self):
        X = self.in_nodes[0].value
        W = self.in_nodes[1].value
        b = self.in_nodes[2].value
        self.value = np.dot(X, W) + b
        # inputs = self.in_nodes[0].value
        # weights = self.in_nodes[1].value
        # bias = self.in_nodes[2].value
        # multi =0
        # for x,y in zip(inputs,weights):
        #     multi = multi+x*y
        # self.value = multi+bias
    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the in_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.out_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
             # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.in_nodes[0]] += np.dot(grad_cost, self.in_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.in_nodes[1]] += np.dot(self.in_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.in_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)
class Sigmoid(node):
    def __init__(self, f):
        # Pass the node to the parent class (node) correctly
        node.__init__(self, [f])  # Correct the argument passed to the parent constructor

    def _sigmoid(self, x):
        return 1./(1.+np.exp(-x))

    def forward(self):
        input_value = self.in_nodes[0].value
        self.value = self._sigmoid(input_value)
    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.out_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.in_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
           
class MSE(node):
    def __init__(self, y,a):
        node.__init__(self, [y,a])
    
    def forward(self):
        y = self.in_nodes[0].value.reshape(-1, 1)
        a = self.in_nodes[1].value.reshape(-1, 1) #将其转为一列数据
        self.m = self.in_nodes[0].value.shape[0]

        self.diff = y - a 
        self.value = np.mean(self.diff**2) #求这列数据的差方的平均值（mean）
    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes
        are not a concern.
        """
        self.gradients[self.in_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.in_nodes[1]] = (-2 / self.m) * self.diff
def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.out_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.out_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in graph:
        n.forward()

    for n in graph[::-1]:  #意思是从反方向遍历
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]
