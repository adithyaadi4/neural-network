import numpy as np  
def relu(x):
    return np.maximum(0,x)
def relu_gradient(x):
    return (x>0).astype(float)
def softmax(x):
    e=np.exp(x-max(x))
    return e/e.sum()
