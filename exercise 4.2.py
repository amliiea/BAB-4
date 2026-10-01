#Example 4.2 Multiple Layer Perceptron: three-input OR
from itertools import product

from sklearn.neural_network import MLPClassifier

X = [list(values) for values in product([0., 1.], repeat=3)]
y = [int(any(values)) for values in X]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
					hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print(clf.predict([[1., 0., 0.], [0., 0., 0.]]))
print([coef.shape for coef in clf.coefs_])
