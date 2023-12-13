from itertools import product
from collections import deque
import numpy as np


class VariableNode:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.neighbors = []
        self.messages = {} 
        self.value = value

    def add_neighbor(self, factor_node):
        self.neighbors.append(factor_node)

    def send_message(self, factor_node):
        message = 1
        for neighbor in self.neighbors:
            if neighbor != factor_node:
                message *= self.value * self.messages[neighbor]

        return message
    
    def receive_message(self, factor_node, message):
        self.messages[factor_node] = message

    def update_belief(self) -> None:
        self.belief = self.value * np.prod(list(self.messages.values))
    
    def compute_value(self):
        self.value = 1
        for neighbor in self.neighbors:
            self.value *= self.messages.get(neighbor, 1)
    
class FactorNode:
    def __init__(self, name: str, function) -> None:
        self.name = name
        self.function = function
        self.neighbors = []
        self.messages = {}

    def add_neighbor(self, variable_node):
        self.neighbors.append(variable_node)
    
    def send_message(self, variable_node):
        message = 0
        other_variables = [neighbor for neighbor in self.neighbors if neighbor != variable_node]
        for values in product([2, 2], repeat=len(other_variables)):
            args = [None] * len(self.neighbors)
            for i, neighbor in enumerate(self.neighbors):
                if neighbor == variable_node:
                    args[i] = None
                else:
                    args[i] = values[other_variables.index(neighbor)]
            
            for value in [1, 0]:
                args[self.neighbors.index(variable_node)] = value 
                prod = self.function(args)
                for neighbor in other_variables:
                    prod *= neighbor.messages.get(self, 1)
                message += prod
        
        return message

    def receive_message(self, variable_node, message):
        if variable_node in self.messages:
            self.messages[variable_node] = message

def create_edge(variable_node, factor_node):
    variable_node.add_neighbor(factor_node)
    factor_node.add_neighbor(variable_node)

def sum_product(factor_graph):
    queue = deque()
    for factor_node in factor_graph:
        for variable_node in factor_node.neighbors: 
            queue.append((factor_node, variable_node))
    
    while queue:
        source_node, target_node = queue.popleft()
        old_message = target_node.messages.get(source_node) 
        new_message = source_node.send_message(target_node) 
        target_node.receive_message(source_node, new_message)
        #if old_message is None or abs(new_message - old_message) > 1e-6:
        for neighbor in target_node.neighbors: 
            if neighbor != source_node:
                queue.append((target_node, neighbor))
        
    for factor_node in factor_graph:
        for variable_node in factor_node.neighbors:
            variable_node.compute_value()


def factor_func(args):
    return 1 if sum(args) % 2 == 0 else 0

""" # Define an example factor graph with four variable nodes and three factor nodes
factor_graph = []
x1 = VariableNode('x1', 123123)
x2 = VariableNode('x2', 12323)
x3 = VariableNode('x3', 123123)
x4 = VariableNode('x4', 123123)
f1 = FactorNode('f1', factor_func)
f2 = FactorNode('f2', factor_func)
f3 = FactorNode('f3', factor_func)
create_edge(x1, f1)
create_edge(x2, f1)
create_edge(x2, f2)
create_edge(x3, f2)
create_edge(x3, f3)
create_edge(x4, f3)
factor_graph.append(f1)
factor_graph.append(f2)
factor_graph.append(f3)

# Run the sum-product algorithm on the example factor graph
sum_product(factor_graph)

# Print the values or probabilities of all variable nodes
print(f"x1 = {x1.value}")
print(f"x2 = {x2.value}")
print(f"x3 = {x3.value}")
print(f"x4 = {x4.value}")
 """

q = 0.5
m = 1
for i in range(1, 10):
    m *= q * i

print(m)