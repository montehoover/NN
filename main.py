import struct
import numpy as np
import pickle

import time

DIGITS = 10
HIDDEN_UNITS = 10
# IMAGES = LABELS = 60000
# COMPONENTS = 50
LRN_RATE = 0.1
ALPHA = 0.1
GAMMA = 0.1
MINI_BATCH = 10
EPOCHS = 1

PICKLE = "nn6.pickle"


def main():
    train_nn()
    test_nn()


def train_nn():
    print("Starting training...")
    start = time.time()
    with open('MNIST_PCA/train-labels.idx1-ubyte', 'rb') as f:
        magic_number = f.read(4)
        num_labels_bytes = f.read(4)
        num_labels = struct.unpack('>i', num_labels_bytes)[0]
        labels_as_ints = np.fromfile(f, dtype=np.uint8)

        # Turn labels into one-hot vectors: Create empty matrix where each row is a one-hot vector of length DIGITS
        # For each row, assign at the index of the label digit to 1. (I.e. if first label is 7, labels[0][7] = 1)
        labels = np.zeros((num_labels, DIGITS), dtype=np.uint8)
        labels[np.arange(num_labels), labels_as_ints] = 1

    with open('MNIST_PCA/train-images-pca.idx2-double', 'rb') as f:
        magic_number = f.read(4)
        num_images_bytes = f.read(4)
        num_components_bytes = f.read(4)
        num_images = struct.unpack('>i', num_images_bytes)[0]
        num_components = struct.unpack('>i', num_components_bytes)[0]
        images = np.fromfile(f, dtype='>d').reshape((num_images, num_components))

    nn = NN(num_components, DIGITS, HIDDEN_UNITS)
    nn.backprop(images, labels)
    with open(PICKLE, 'wb') as f:
        pickle.dump(nn, f)

    print("Finished training in {} seconds".format(time.time() - start))


def test_nn():
    print("Starting testing...")
    start = time.time()
    with open('MNIST_PCA/t10k-labels.idx1-ubyte', 'rb') as f:
        magic_number = f.read(4)
        num_labels_bytes = f.read(4)
        num_labels = struct.unpack('>i', num_labels_bytes)[0]
        labels_as_ints = np.fromfile(f, dtype=np.uint8)

        # # Turn labels into one-hot vectors: Create empty matrix where each row is a one-hot vector of length DIGITS
        # # For each row, assign at the index of the label digit to 1. (I.e. if first label is 7, labels[0][7] = 1)
        # labels = np.zeros((num_labels, DIGITS), dtype=np.uint8)
        # labels[np.arange(num_labels), labels_as_ints] = 1

    with open('MNIST_PCA/t10k-images-pca.idx2-double', 'rb') as f:
        magic_number = f.read(4)
        num_images_bytes = f.read(4)
        num_components_bytes = f.read(4)
        num_images = struct.unpack('>i', num_images_bytes)[0]
        num_components = struct.unpack('>i', num_components_bytes)[0]
        images = np.fromfile(f, dtype='>d').reshape((num_images, num_components))

    with open(PICKLE, 'rb') as f:
        nn = pickle.load(f)

    assert num_images == num_labels
    errors = 0
    for i in range(num_images):
        # print(labels_as_ints[i], nn.classify(images[i]))
        if labels_as_ints[i] != nn.classify(images[i]):
            errors += 1
    print(errors / num_images)

    print("Finished testing in {} seconds".format(time.time() - start))



class NN():
    def __init__(self, num_input: int, num_output: int, num_hidden: int):
        # Weights from network input to hidden layer units; each row in matrix corresponds to weights for a single unit
        self.wh = np.random.uniform(-0.05, 0.05, (num_hidden, num_input + 1))
        # Weights from hidden layer to output layer units; each row in matrix corresponds to weights for a single unit
        self.wk = np.random.uniform(-0.05, 0.05, (num_output, num_hidden + 1))


    def backprop(self, examples: np.ndarray, labels: np.ndarray) -> None:
        for y in range(EPOCHS):
            batch_of_dwhs = np.empty((*self.wh.shape, 0))
            batch_of_dwks = np.empty((*self.wk.shape, 0))
            # Run one epoch
            for z in range(len(examples)):
                # Propogate input values through
                o_h_vector = self.get_h_layer_output(examples[z])
                o_k_vector = self.get_k_layer_output(o_h_vector)

                if z % 1000 == 0:
                    print(z, o_k_vector)

                # Propogate errors back
                delta_ks = [self.delta_k(o_k, labels[z][i]) for i, o_k in enumerate(o_k_vector)]
                # To calculate delta_h, we need vector of weights that go from that h-unit to each output unit
                delta_hs = [self.delta_h(o_h, self.wk[:, i], delta_ks) for i, o_h in enumerate(o_h_vector)]

                # Add 1.0 as first element in x vector to be paired with w_0
                xs = np.insert(examples[z], 0, 1.0)
                o_h_vector = np.insert(o_h_vector, 0, 1.0)
                # Turn into horizontal vector and vertical vectors in preparation for element-wise multiplication into matrices
                xs = xs.reshape(1, *xs.shape)
                o_h_vector = np.array(o_h_vector).reshape(1, len(o_h_vector))
                delta_ks = np.array(delta_ks).reshape(len(delta_ks), 1)
                delta_hs = np.array(delta_hs).reshape(len(delta_hs), 1)

                dwh = LRN_RATE * (delta_hs * xs)
                np.append(batch_of_dwhs, dwh.reshape(*dwh.shape, 1), axis=2)

                dwk = LRN_RATE * (delta_ks * o_h_vector)
                np.append(batch_of_dwks, dwk.reshape(*dwk.shape, 1), axis=2)


                # If we have finished collecting errors for a mini-batch, update the weights according to gradient descent
                if (z + 1) % MINI_BATCH == 0:
                    # Sum the corresponding weights for the batch; axis 2 holds each weight matrix as a slice
                    self.wh = self.wh + batch_of_dwhs.sum(axis=2)
                    self.wk = self.wk + batch_of_dwks.sum(axis=2)


    def classify(self, x):
        o_h_vector = self.get_h_layer_output(x)
        o_k_vector = self.get_k_layer_output(o_h_vector)

        classified_digit = np.argmax(o_k_vector)
        return classified_digit


    def delta_w(self, delta_j_vector, xji_vector):
        return LRN_RATE * delta_j_vector.dot(xji_vector)

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
        return [self.sigmoid_unit_output(w, x) for w in self.wh]

    def get_k_layer_output(self, x: np.ndarray) -> np.ndarray:
        """
        Get vector of values from output units.
        Each output unit receives the same vector of input from hidden layer units and applies a unique vector of
        weights to its sigmoid unit calculation
        :param x:
        :return: vector of all values from the k-layer output units
        """
        return [self.sigmoid_unit_output(w, x) for w in self.wk]

    def sigmoid_unit_output(self, w: np.ndarray, x: np.ndarray) -> float:
        # Insert x_0 = 1 to x vector to correspond with w_0
        x = np.insert(x, 0, 1)
        net = w.dot(x)
        return self.sigmoid(net)

    def sigmoid(self, x: float) -> float:
        if x > 650:
            x = 650
        elif x < -650:
            x = -650
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    main()