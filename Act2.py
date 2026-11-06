from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot

x, y = make_blobs(n_samples=1000, centers=2, random_state=1)

print(x.shape, y.shape)

Counter = Counter(y)
print(Counter)

for i in range(10):
    print(x[i], y[i])

for label, _ in Counter.items():
    row_ix = where (y == label)[0]
    pyplot.scatter(x[row_ix, 0], x[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()