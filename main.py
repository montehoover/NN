import struct

import numpy
import numpy as np

DIGITS = 10
HIDDEN_UNITS = 10
IMAGES = LABELS = 60000
COMPONENTS = 50

def main():
    with open('MNIST_PCA/train-labels.idx1-ubyte', 'rb') as f:
        magic_number = f.read(4)
        num_labels = f.read(4)
        assert LABELS == struct.unpack('>i', num_labels)[0]
        labels_as_ints = np.fromfile(f, dtype=np.uint8)

        # Turn labels into one-hot vectors: Create empty matrix where each row is a one-hot vector of length DIGITS
        # For each row, assign at the index of the label digit to 1. (I.e. if first label is 7, labels[0][7] = 1)
        labels = np.zeros((LABELS, DIGITS), dtype=np.uint8)
        labels[np.arange(LABELS), labels_as_ints] = 1


    with open('MNIST_PCA/train-images-pca.idx2-double', 'rb') as f:
        magic_number = f.read(4)
        num_images = f.read(4)
        num_components = f.read(4)
        assert IMAGES == struct.unpack('>i', num_images)[0] and COMPONENTS == struct.unpack('>i', num_components)[0]
        images = np.fromfile(f, dtype='>d').reshape((IMAGES, COMPONENTS))

    nn = NN(COMPONENTS, DIGITS, HIDDEN_UNITS)
    l = list(images)
    print(l[0])
    o = nn.prop_input(l[0])
    print(o)


def sigmoid_unit_output(w: numpy.ndarray, x: numpy.ndarray) -> float:
    net = w.dot(x)
    return sigmoid(net)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class NN():
    def __init__(self, num_input: int, num_output: int, num_hidden: int):
        self.hidden_weights = np.random.uniform(-0.05, 0.05, (num_hidden, num_input + 1))
        self.output_weights = np.random.uniform(-0.05, 0.05, (num_output, num_hidden + 1))

    def prop_input(self, x: numpy.ndarray) -> numpy.ndarray:
        x = np.insert(x, 0, 1)
        o = [sigmoid_unit_output(w, x) for w in self.hidden_weights]
        return o


if __name__ == '__main__':
    main()