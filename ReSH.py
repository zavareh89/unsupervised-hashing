# ref : "Reversed Spectral Hashing" paper (2018)
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K_
#from tensorflow.linalg import matmul


def create_dense_network(input_dim, hidden_dim, output_dim):
    input_ = keras.layers.Input(input_dim)
    hidden = keras.layers.Dense(hidden_dim, activation="sigmoid")(input_)
    output = keras.layers.Dense(output_dim, activation="sigmoid")(hidden)
    model = keras.models.Model(inputs=[input_], outputs=[output], name='ReSH')
    print('model was created')
    model.summary()
    return model


def pairwise_dist(A, B):
    """
    Computes pairwise Euclidean distances between each elements of A
        and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), axis=1)
    nb = tf.reduce_sum(tf.square(B), axis=1)

    # na as a column and nb as a row vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, transpose_a=False,
                                              transpose_b=True) + nb, 0.0))
    return D


def cost_ReSH(X, Y, X_sub, Y_sub, p=8, sigma=1):
    n, r = X.shape[0], X_sub.shape[0]
    # input distances
    input_dist = tf.math.pow(pairwise_dist(X, X_sub), p)
    # output distances
    output_dist = tf.math.pow(pairwise_dist(Y, Y_sub), 2)
    # compute cost
    thr_param = tf.multiply(2, tf.pow(sigma, 2))
    E = tf.reduce_sum(tf.multiply(input_dist, tf.exp(-output_dist / thr_param)))
    return E / (n * r)


def Rprop_update(params, grads, prev_grads, prev_weight_deltas, deltas, Rprop_params):
    # some parts of this implementation are from the following repository:
    #   https://github.com/ntnu-ai-lab/RProp
    d_min = Rprop_params['delta_min']
    d_max = Rprop_params['delta_min']
    scale_down = Rprop_params['scale_down']
    scale_up = Rprop_params['scale_up']

    # update parameters using Rprop algorithm (see the original 1993 paper)
    weight_deltas = []
    for param, grad, prev_grad, prev_weight_delta, delta in zip(params,
                                                                grads, prev_grads,
                                                                prev_weight_deltas,
                                                                deltas):
        # equation 4 (equation numbers are from the 1993 paper)
        prod_sign = tf.multiply(tf.math.sign(grad), tf.math.sign(prev_grad))
        delta.assign(K_.switch(
            K_.greater(prod_sign, 0),
            K_.minimum(delta * scale_up, d_max),
            K_.switch(K_.less(prod_sign, 0), K_.maximum(delta * scale_down, d_min), delta)
        ))

        # equation 5
        weight_delta = tf.multiply(-tf.math.sign(grad), delta)

        # equation 7
        weight_delta = K_.switch(K_.less(prod_sign, 0), -prev_weight_delta, weight_delta)

        # equation 6 (update model variables)
        param.assign(param + weight_delta)

        # update previous weight delta variable
        prev_weight_delta.assign(weight_delta)

        # reset gradient to 0 if gradient sign changed (so that we do
        # not "double punish", see paragraph after equation 7)
        prev_grad.assign(K_.switch(K_.less(prod_sign, 0), tf.zeros_like(grad), grad))

    return deltas, prev_grads, prev_weight_deltas


def train_ReSH(train_features, model, K=16, n_epochs=70, r=0.05, sigma=None,
               p=8, Rprop_params=None):
    """ Reverse Spectral hashing (ReSH)
        train_features: shape is (n_samples,n_features)
        model: the network which is going to be trained.
        K: Number of bits for each binary code
        n_epochs: number of epochs
        r: the fraction of total train examples which is randomly chosen in
            each epoch to approximate the loss function.
        sigma: the thresholding parameter
        p: the power of norm used in loss function (p-norm). See equation 7.
        Rprop_params: This is a dictionary containing the The resilient
            backpropagation (Rprop) paramters. The keys are:
                delta_0: initial weight change
                delta_min: minimum weight change
                delta_max: maximum weight change
                scale_down: decrement to weight change
                scale_up: increment to weight change

        outputs:
            model: trained model
            cost_values: cost values with shapr (n_epochs,)
            outputs: the neural network's outputs in the last epoch. Its
                shape is (n_train, K) which n_train is number of train examples.
    """
    n_train = train_features.shape[0]
    if r > 1 or r < 0:
        raise ValueError('"r" value must be between 0 and 1')
    r = int(np.ceil(r * n_train))
    train_features = tf.constant(train_features, dtype=np.float32)
    if sigma is None:
        sigma = tf.constant(0.2 * np.sqrt(K), dtype=np.float32)
    cost_values = np.zeros((n_epochs,), dtype=np.float32)

    # set default values for unspecified Rprop parameters
    if Rprop_params is None:
        Rprop_params = {}
    if not isinstance(Rprop_params, dict):
        raise TypeError('"Rprop_params" must be a dictionary')
    Rprop_params['delta_0'] = tf.constant(Rprop_params.get('delta_0', 0.07),
                                          dtype=tf.float32)
    Rprop_params['delta_min'] = tf.constant(Rprop_params.get('delta_min', 1e-6),
                                            dtype=tf.float32)
    Rprop_params['delta_max'] = tf.constant(Rprop_params.get('delta_max', 50),
                                            dtype=tf.float32)
    Rprop_params['scale_down'] = tf.constant(Rprop_params.get('scale_down', 0.5),
                                             dtype=tf.float32)
    Rprop_params['scale_up'] = tf.constant(Rprop_params.get('scale_up', 1.2),
                                           dtype=tf.float32)

    # initialize old_grads, deltas, and prev_weight_deltas for Rprop
    shapes = [v.shape for v in model.trainable_variables]
    deltas = [tf.Variable(tf.ones(shape) * Rprop_params['delta_0']) for shape in shapes]
    prev_grads = [tf.Variable(tf.zeros(shape)) for shape in shapes]
    prev_weight_deltas = [tf.Variable(tf.zeros(shape)) for shape in shapes]

    # main loop (we compute gradient for the sub-sample and compute output
    #   for all training examples)
    for epoch in range(1, n_epochs + 1):
        # random index of sub-sample
        idx = np.random.choice(n_train, r, replace=False)
        sub_sample = train_features.numpy()[idx, :]
        sub_sample = tf.constant(sub_sample, dtype=np.float32)
        print(f"Epoch {epoch}/{n_epochs}", end="")
        # compute output for all examples
        output_all = model.predict(train_features, batch_size=512)
        output_all = tf.constant(output_all, dtype=np.float32)
        # compute gradient for sub-sample
        with tf.GradientTape() as tape:
            output_sub = model(sub_sample, training=True)
            # for cost definition, see equation 10 of the paper.
            cost = cost_ReSH(train_features, output_all, sub_sample,
                             output_sub, p=p, sigma=sigma)
        cost_values[epoch - 1] = cost.numpy()
        print(f"  loss={cost_values[epoch - 1]}")
        gradients = tape.gradient(cost, model.trainable_variables)
        # update parameters usin Rprop algorithm
        deltas, prev_grads, prev_weight_deltas = Rprop_update(
            model.trainable_variables, gradients, prev_grads,
            prev_weight_deltas, deltas, Rprop_params
        )

    return model, cost_values, output_all.numpy()
