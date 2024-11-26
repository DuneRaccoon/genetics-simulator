import random
import math
from typing import List

class NeuralNetwork:
    
    def __init__(self, layer_sizes: List[int], weights=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # Initialize weights and biases
        if weights:
            self.weights = weights
        else:
            self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        for i in range(self.num_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            # Initialize weights and biases with small random values
            layer_weights = {
                'weights': [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)],
                'biases': [random.uniform(-1, 1) for _ in range(output_size)]
            }
            weights.append(layer_weights)
        return weights

    def forward(self, inputs: List[float]):
        activations = inputs
        for layer in self.weights:
            next_activations = []
            for w_row, b in zip(layer['weights'], layer['biases']):
                # Compute weighted sum
                z = sum(w * a for w, a in zip(w_row, activations)) + b
                # Apply activation function (sigmoid)
                a = 1 / (1 + math.exp(-z))
                next_activations.append(a)
            activations = next_activations
        return activations

    def get_genome(self):
        # Flatten weights and biases into a single list
        genome = []
        for layer in self.weights:
            for w_row in layer['weights']:
                genome.extend(w_row)
            genome.extend(layer['biases'])
        return genome

    @staticmethod
    def from_genome(layer_sizes: List[int], genome: List[float]):
        weights = []
        index = 0
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            layer_weights = {
                'weights': [],
                'biases': []
            }
            for _ in range(output_size):
                w_row = genome[index:index + input_size]
                index += input_size
                layer_weights['weights'].append(w_row)
            biases = genome[index:index + output_size]
            index += output_size
            layer_weights['biases'] = biases
            weights.append(layer_weights)
        return NeuralNetwork(layer_sizes, weights)
