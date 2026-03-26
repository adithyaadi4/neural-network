from weights import initialize_weights
from train import train
from predict import predict, test_all

# ── step 1: initialize weights ──
initialize_weights(seed=42)

# ── step 2: train ──
train(epochs=1000, lr=0.1)

# ── step 3: test all lines ──
test_all()

# ── step 4: test your own inputs ──
print("\nTESTING CUSTOM INPUTS:")
print("─" * 40)

# test each line
predict([1,1,1, 0,0,0, 0,0,0])   # Htop
predict([0,0,0, 1,1,1, 0,0,0])   # Hmiddle
predict([0,0,0, 0,0,0, 1,1,1])   # Hdown
predict([1,0,0, 1,0,0, 1,0,0])   # Vleft
predict([0,1,0, 0,1,0, 0,1,0])   # Vmiddle
predict([0,0,1, 0,0,1, 0,0,1])   # Vright

# test noisy input
print("NOISY INPUT TEST:")
print("─" * 40)
predict([1,1,1, 0,0,0, 0,1,0])   # Htop with one wrong pixel

# interactive test
print("\nINTERACTIVE TEST:")
print("─" * 40)
print("Enter 9 pixels (0 or 1) separated by spaces:")
print("Example: 1 1 1 0 0 0 0 0 0")
user_input = input("→ ")
pixels = list(map(int, user_input.split()))
predict(pixels)