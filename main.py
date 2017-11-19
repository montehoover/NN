import struct
import numpy as np
import pickle

import time

DIGITS = 10
HIDDEN_UNITS = 10
# IMAGES = LABELS = 60000
# COMPONENTS = 50
LRN_RATE = 0.1
MINI_BATCH = 10
EPOCHS = 1

PICKLE = "nn5.pickle"


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
        labels = to_one_hot_vector(labels_as_ints)

        # Turn labels into one-hot vectors: Create empty matrix where each row is a one-hot vector of length DIGITS
        # For each row, assign at the index of the label digit to 1. (I.e. if first label is 7, labels[0][7] = 1)
        # labels = np.zeros((num_labels, DIGITS), dtype=np.uint8)
        # labels[np.arange(num_labels), labels_as_ints] = 1

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
        labels = to_one_hot_vector(labels_as_ints)

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

    a = squared_loss()
    # b = oh_over_one_loss()
    print(errors / num_images)
    print("Finished testing in {} seconds".format(time.time() - start))



def to_one_hot_vector(x: np.ndarray) -> np.ndarray:
    """
    Turn labels into one-hot vectors: Create empty matrix where each row is a one-hot vector of length DIGITS
    For each row, assign at the index of the label digit to 1. (I.e. if first label is 7, labels[0][7] = 1)
    :param x: 1D np array
    :return:  2d np array
    """
    one_hot = np.zeros((len(x), DIGITS), dtype=np.uint8)
    one_hot[np.arange(len(x)), x] = 1
    return one_hot


def squared_loss(nn: object, test_images: np.ndarray, test_labels: np.ndarray) -> float:
    """
    :param nn:
    :param test_images: 2D matrix of images x image features
    :param test_labels: 2D matrix of labels x one-hot representation
    """
    return sum([single_squared_loss(nn, *i_l_tuple) for i_l_tuple in zip(test_images, test_labels)])

def single_squared_loss(nn: object, image_v: np.ndarray, label_onehot: np.ndarray) -> float:
    """
    Measurement of NN's classification of an image against a known label
    :param nn:
    :param image_v: 1D vector of image features (pixels or PCA, etc.)
    :param label_onehot: 1D vector (one-hot) labeling matching the image
    :return: Scalar value that is the Squared Loss for the single image
    """
    classified_onehot = nn.get_output(image_v)
    differences = label_onehot - classified_onehot
    squares = np.square(differences)
    sum = np.sum(squares)
    return sum


class NN():
    def __init__(self, num_input: int, num_output: int, num_hidden: int):
        # Weights from network input to hidden layer units; each row in matrix corresponds to weights for a single unit
        self.w_hn = np.random.uniform(-0.05, 0.05, (num_hidden, num_input + 1))
        # Weights from hidden layer to output layer units; each row in matrix corresponds to weights for a single unit
        self.w_kh = np.random.uniform(-0.05, 0.05, (num_output, num_hidden + 1))


    def backprop(self, examples: np.ndarray, labels: np.ndarray) -> None:
        for y in range(EPOCHS):
            batch_of_xhs = []
            batch_of_xks = []
            batch_of_dhs = []
            batch_of_dks = []
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
                delta_hs = [self.delta_h(o_h, self.w_kh[:, i], delta_ks) for i, o_h in enumerate(o_h_vector)]

                # Collect errors and values for mini-batch
                # Each is a list of lists, where each list is a row appended to the matrix. The rows in the matrix are
                # values corresponding to a single training example (50 values for a single image, HIDDEN_UNITS # of o_h
                # outputs, HIDDEN_UNITS # of delta_hs, and 10 delta_ks for each of the 10 k-layer outputs).
                batch_of_xhs.append(examples[z])
                batch_of_xks.append(o_h_vector)
                batch_of_dhs.append(delta_hs) # matrix where the rows are for each of the hidden units, and the columns are for all 10 of the batch for the same hidden unit
                batch_of_dks.append(delta_ks)

                # If we have finished collecting errors for a mini-batch, update the weights according to gradient descent
                if (z + 1) % MINI_BATCH == 0:
                    # turn lists of lists into matrices:
                    batch_of_xhs = np.array(batch_of_xhs)
                    batch_of_xks = np.array(batch_of_xks)
                    batch_of_dhs = np.array(batch_of_dhs)
                    batch_of_dks = np.array(batch_of_dks)

                    # for each set of 50 weights going to HIDDEN_UNITS # of units (say 5 or 10)
                    for j, weights in enumerate(self.w_hn):
                        # for each of the 50 weights in that set
                        for i, weight in enumerate(weights):
                            if i == 0:
                                # to get paired with w_0
                                x_ji_vector = np.ones((MINI_BATCH,1))
                            else:
                                # use -1 because there are 50 x's in each example, and 51 weights because of w_0
                                x_ji_vector = batch_of_xhs[:, i-1]
                            # Update all 50 weights for each hidden unit, then update all 50 for the next one, etc.
                            # here delta_hs[j] stays the same and x_ji keeps changing
                            # for a mini batch, we want the x_ji from the first ten examples: so x_ji from example1 through
                            # example10, and we want delta_h for that hidden unit for the first ten examples.
                            # So we should have a dot-product of two 10-item vectors here. If we have delta_h's in a list
                            # for all 5 hidden units, and append those lists as rows in a matrix, then we can get that
                            # 10-item vector by taking the columns of that matrix.
                            weights[i] = weights[i] + self.delta_w(batch_of_dhs[:, j], x_ji_vector)

                    # for each set of HIDDEN_UNITS # of weights going to the 10 outputs
                    for j, weights in enumerate(self.w_kh):
                        # for each of the HIDDEN_UNITS # of weights (say 5 or 10)
                        for i, weight in enumerate(weights):
                            if i == 0:
                                x_ji_vector = np.ones((MINI_BATCH,1))
                            else:
                                x_ji_vector = batch_of_xks[:, i-1]
                            # Update the 5 weights for each output unit, then update all 5 for the next one, etc.
                            weights[i] = weights[i] + self.delta_w(batch_of_dks[:, j], x_ji_vector)

                    # Clear the contents of these matrices so they're ready for the next mini-batch
                    batch_of_xhs = []
                    batch_of_xks = []
                    batch_of_dhs = []
                    batch_of_dks = []


    def classify(self, x: np.ndarray) -> int:
        """
        :param x: 1D vector of input values (from a single image)
        :return: The NN's classification value of that image
        """
        classified_onehot = self.get_output(x)
        classified_digit = np.argmax(classified_onehot)
        return classified_digit

    def get_output(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: 1D vector of input values (from a single image)
        :return: 1D vector (one-hot) of the NN's classification value of that image
        """
        o_h_vector = self.get_h_layer_output(x)
        return self.get_k_layer_output(o_h_vector)


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
        Get vector of output values from hidden layer units.
        Each hidden layer unit receives the same vector of input values (x) and applies a unique vector of
        weights (w) to its sigmoid unit calculation.
        :param x: 1D vector of input values (from a single image)
        :return: 1D vector of hidden layer output values
        """
        return [self.sigmoid_unit_output(w, x) for w in self.w_hn]

    def get_k_layer_output(self, x: np.ndarray) -> np.ndarray:
        """
        Get vector of values from output units.
        Each output unit receives the same vector of input from hidden layer units and applies a unique vector of
        weights to its sigmoid unit calculation
        :param x:
        :return: vector of all values from the k-layer output units
        """
        return [self.sigmoid_unit_output(w, x) for w in self.w_kh]

    def sigmoid_unit_output(self, w: np.ndarray, x: np.ndarray) -> float:
        """
        :param w: 1D vector of weights
        :param x: 1D vector of inputs
        :return: Scalar value output
        """
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