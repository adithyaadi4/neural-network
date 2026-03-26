import numpy as np

# global weights
W1 = None
W2 = None

def initialize_weights(seed=42):
    global W1, W2
    np.random.seed(seed)
    W1 = np.random.randn(9, 6) * 0.1  # 9 inputs  → 6 hidden
    W2 = np.random.randn(6, 6) * 0.1  # 6 hidden  → 6 outputs
    print("Weights initialized!")
    print(f"W1 shape: {W1.shape}  (9 inputs × 6 hidden)")
    print(f"W2 shape: {W2.shape}  (6 hidden × 6 outputs)")

def get_weights():
    return W1, W2

def set_weights(new_W1, new_W2):
    global W1, W2
    W1 = new_W1
    W2 = new_W2
initialize_weights(seed=42)

    # get weights
W1, W2 = get_weights()
print("\nW1 ",W1.shape)
print("         h1      h2      h3      h4      h5      h6")
for i, row in enumerate(W1):
        print(f"p{i+1}  =  {row}")

print("\nW2 (6×6):")
print("         o1      o2      o3      o4      o5      o6")
for i, row in enumerate(W2):
        print(f"h{i+1}  =  {row}")  
