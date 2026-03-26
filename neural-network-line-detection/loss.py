import numpy as np

def cross_entropy_loss(output, label):
    # how wrong is the prediction
    # high loss = very wrong
    # low loss  = nearly correct
    return -np.log(output[label] + 1e-8)

def total_loss(outputs, labels):
    # average loss across all examples
    total = 0
    for output, label in zip(outputs, labels):
        total += cross_entropy_loss(output, label)
    return total / len(labels)