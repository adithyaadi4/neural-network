A simple project built to understand neural networks. a neural network built from scratch to detect lines.
Htop        Hmiddle     Hdown
1  1  1     0  0  0     0  0  0
0  0  0     1  1  1     0  0  0
0  0  0     0  0  0     1  1  1

Vleft       Vmiddle     Vright
1  0  0     0  1  0     0  0  1
1  0  0     0  1  0     0  0  1
1  0  0     0  1  0     0  0  1

 detects all these lines.learns through training where algorithms like relu,softmax have been used.

 network architecture
 Input Layer  → 9 neurons   (one per pixel)
Hidden Layer → 6 neurons   (one per line type)
Output Layer → 6 neurons   (Htop, Hmiddle, Hdown, Vleft, Vmiddle, Vright)

W1 = (9×6) = 54 weights
W2 = (6×6) = 36 weights
Total       = 90 weights

how it works?
Forward pass:
Input → weighted sum → ReLU → weighted sum → Softmax → Prediction

Backward pass:
Loss → output gradient → W2 gradient → hidden gradient → W1 gradient → update weights

Repeat 1000 epochs until network learns!
