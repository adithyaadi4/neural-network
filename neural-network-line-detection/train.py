from data import inputs, labels
from forward import forward
from loss import cross_entropy_loss
from backward import backward

def train(epochs=1000, lr=0.1):
    print("TRAINING:")
    print("─" * 40)

    for epoch in range(epochs + 1):

        total_loss = 0

        for i in range(len(inputs)):
            x     = inputs[i]
            label = labels[i]

            # forward pass
            z1, h, z2, output = forward(x)

            # calculate loss
            total_loss += cross_entropy_loss(output, label)

            # backward pass + weight update
            backward(x, z1, h, output, label, lr)

        # print every 100 epochs
        if epoch % 100 == 0:
            avg_loss = total_loss / len(inputs)
            print(f"Epoch {epoch:4d} → Loss: {avg_loss:.4f}")

    print("─" * 40)
    print("Training complete!")