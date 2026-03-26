import numpy as np
from activations import relu_gradient
from weights import get_weights, set_weights

def backward(x, z1, h, output, label, lr=0.1):
    W1, W2 = get_weights()

    # ── step 1: output gradient ──
    # how wrong is each output neuron
    d_output = output.copy()
    d_output[label] -= 1        # predicted - actual

    # ── step 2: W2 gradient ──
    # how much did each W2 weight contribute to error
    dW2 = np.outer(h, d_output)

    # ── step 3: hidden gradient ──
    # how much did each hidden neuron contribute to error
    dh = np.dot(W2, d_output)
    dh *= relu_gradient(z1)     # blocked if neuron was silent

    # ── step 4: W1 gradient ──
    # how much did each W1 weight contribute to error
    dW1 = np.outer(x, dh)

    # ── step 5: update weights ──
    W1_new = W1 - lr * dW1
    W2_new = W2 - lr * dW2
    set_weights(W1_new, W2_new)

    return dW1, dW2