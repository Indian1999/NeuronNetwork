import numpy as np # terminálba: pip install numpy

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.weights = np.random.random((3, 1)) * 2 - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, train_x, train_y, epoch):
        for i in range(epoch):
            output = self.predict(train_x)
            error = train_y - output
            adjust = np.dot(train_x.T, error * self.sigmoid_derivative(output))
            self.weights += adjust

    def predict(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output



model = NeuralNetwork()
print("Kezdeti súly értékek:", model.weights)

train_x = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])
train_y = np.array([[0, 1, 1, 0]]).T # 4 sor, 1 oszlop
model.train(train_x, train_y, 1000)

print("Tanítás után súly értékek:", model.weights)

test_x = np.array([ [0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1],
                    ])
print("#" *50)
print(model.predict(test_x))























"""
class Vector2():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x},{self.y})"
    
    def invert(self):
        self.x *= -1
        self.y *= -1

    def inverted(self):
        return Vector2(self.x * -1, self.y * -1)

"""