#Example 4.1 Neuron.py
from numpy import exp, array, random, dot, ones, hstack

#Set up parameters ===============================================
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
outputs = array([[0, 1, 1, 1]], dtype=float).T
random.seed(1)
weights = 2 * random.random((3, 1)) - 1
inputs_with_bias = hstack([inputs, ones((len(inputs), 1))])

#Define a Single Neuron function =================================
def neuron(input_vector, weights):
	output = 1 / (1 + exp(-(dot(input_vector, weights))))
	return output

#Train the Neuron ================================================
for iteration in range(50000):
	output = neuron(inputs_with_bias, weights)
	weights += dot(inputs_with_bias.T, (outputs - output) * output *
					(1 - output))

#Test the Neuron =================================================
def classify_or(input_vector, weights):
	x = hstack([input_vector, [1.0]])
	value = neuron(x, weights)
	return 1 if value >= 0.5 else 0

for x in inputs:
	print(x.tolist(), '->', classify_or(x, weights))

# Example of single input test
x = array([1, 0])
print('input', x.tolist(), 'output', classify_or(x, weights))
