import pandas as pd
import numpy as np
from activations import relu,softmax
from weights import get_weights
def forward(x):
    W1, W2 = get_weights()
    z1=np.dot(x,W1)
    h=relu(z1)
    z2=np.dot(h,W2)
    output=softmax(z2)
    return z1,h,z2,output