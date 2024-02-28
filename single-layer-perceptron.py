import numpy

class Perceptron(object):
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = numpy.zeros(input_size + 1)
        self.epochs = epochs
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = numpy.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation(z)
        return a
    
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                print("predict(" + str(X[i]) + ") = " + str(y))
                e = d[i] - y
                print("e = " + str(e))
                self.W = self.W + self.lr * e * numpy.insert(X[i], 0, 1)
                print("weight = " + str(self.W))
                print("\n")

if __name__ == '__main__':
    X = numpy.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    d = numpy.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, d)
    print(perceptron.W)
    print(perceptron.predict([0, 0]))
    print(perceptron.predict([0, 1]))
    print(perceptron.predict([1, 0]))
    print(perceptron.predict([1, 1]))