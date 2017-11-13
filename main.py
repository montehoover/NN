import struct
import numpy as np

DIGITS = 10
HIDDEN_UNITS = 5
IMAGES = LABELS = 60000
COMPONENTS = 50
LRN_RATE = 0.1

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
    o_h = nn.get_h_layer_output(images[0])
    print(o_h)
    o_k = nn.get_k_layer_output(o_h)
    print(o_k)
    print(labels[0])

    print(nn.w_hi[0])
    nn.backprop(images, labels)
    print(nn.w_hi[0])
    o_h = nn.get_h_layer_output(images[0])
    print(o_h)
    o_k = nn.get_k_layer_output(o_h)
    print(o_k)



def sigmoid_unit_output(w: np.ndarray, x: np.ndarray) -> float:
    # Insert x_0 = 1 to x vector to correspond with w_0
    x = np.insert(x, 0, 1)
    net = w.dot(x)
    return sigmoid(net)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class NN():
    def __init__(self, num_input: int, num_output: int, num_hidden: int):
        # Weights from network input to hidden layer units; each row in matrix corresponds to weights for a single unit
        self.w_hi = np.random.uniform(-0.05, 0.05, (num_hidden, num_input + 1))
        # Weights from hidden layer to output layer units; each row in matrix corresponds to weights for a single unit
        self.w_kh = np.random.uniform(-0.05, 0.05, (num_output, num_hidden + 1))


    def backprop(self, examples: np.ndarray, labels: np.ndarray) -> None:
        # delta ks from first training example
        z = 0
        o_h_vector = self.get_h_layer_output(examples[z])
        o_k_vector = self.get_k_layer_output(o_h_vector)
        delta_ks = [self.delta_k(o_k, labels[z][i]) for i, o_k in enumerate(o_k_vector)]
        # To calculate delta_h, we need vector of weights that go from that h-unit to each output unit
        delta_hs = [self.delta_h(o_h, self.w_kh[:, i], delta_ks) for i, o_h in enumerate(o_h_vector)]

        j = 0 # which hidden unit
        i = 0 # which weight value
        for j, weights in enumerate(self.w_hi):
            for i, weight in enumerate(weights):
                if i == 0:
                    x_ji = 1
                else:
                    x_ji = examples[z][i - 1]
                weight = weight + self.delta_w(delta_hs[j], x_ji)

        # self.w_hi[j, i] = self.w_hi[j, i] + self.delta_w(delta_hs[j], x_ji)


    def delta_w(self, delta_j, xji):
        return LRN_RATE * delta_j * xji

    def delta_k(self, o_k: float, t_k: int) -> float:
        return o_k * (1 - o_k) * (t_k - o_k)

    def delta_h(self, o_h: float, w_kh_vector: np.ndarray, delta_ks: np.ndarray) -> float:
        """

        :param o_h:
        :param w_kh_vector: vector of all weights from single hidden unit to each output unit, k
        :param delta_ks:
        :return:
        """
        return o_h * (1 - o_h) * w_kh_vector.dot(delta_ks)

    def get_h_layer_output(self, x: np.ndarray) -> np.ndarray:
        """
        Get vector of values from output units.
        Each hidden layer unit receives the same vector of input values and applies a unique vector of
        weights to its sigmoid unit calculation
        :param x:
        :return: vector of all hidden layer output values
        """
        return [sigmoid_unit_output(w, x) for w in self.w_hi]

    def get_k_layer_output(self, x: np.ndarray) -> np.ndarray:
        """
        Get vector of values from output units.
        Each output unit receives the same vector of input from hidden layer units and applies a unique vector of
        weights to its sigmoid unit calculation
        :param x:
        :return: vector of all values from the k-layer output units
        """
        return [sigmoid_unit_output(w, x) for w in self.w_kh]


if __name__ == '__main__':
    main()