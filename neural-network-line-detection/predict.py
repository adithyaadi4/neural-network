import numpy as np
from forward import forward
from data import line_names

def predict(input_pixels):
    x = np.array(input_pixels, dtype=float)

    z1, h, z2, output = forward(x)

    predicted  = np.argmax(output)
    confidence = output[predicted] * 100

    print("\nInput grid:")
    print(f"{int(x[0])}  {int(x[1])}  {int(x[2])}")
    print(f"{int(x[3])}  {int(x[4])}  {int(x[5])}")
    print(f"{int(x[6])}  {int(x[7])}  {int(x[8])}")

    print("\nProbabilities:")
    for i, name in enumerate(line_names):
        bar = "█" * int(output[i] * 50)
        print(f"{name:10s} {output[i]*100:5.1f}%  {bar}")

    print(f"\nPrediction → {line_names[predicted]}  ({confidence:.1f}%)")

def test_all():
    from data import inputs, labels
    correct = 0

    for i in range(len(inputs)):
        x     = inputs[i]
        label = labels[i]

        z1, h, z2, output = forward(x)
        predicted  = np.argmax(output)
        confidence = output[predicted] * 100

        status = "✓" if predicted == label else "✗"
        print(f"{status} {line_names[label]:10s} → {line_names[predicted]:10s}  {confidence:.1f}%")

        if predicted == label:
            correct += 1

    print(f"Accuracy: {correct}/{len(inputs)} = {correct/len(inputs)*100:.1f}%")