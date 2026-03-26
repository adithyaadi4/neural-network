import numpy as np
inputs = np.array([
    [1,1,1, 0,0,0, 0,0,0],  # Htop    → 0
    [0,0,0, 1,1,1, 0,0,0],  # Hmiddle → 1
    [0,0,0, 0,0,0, 1,1,1],  # Hdown   → 2
    [1,0,0, 1,0,0, 1,0,0],  # Vleft   → 3
    [0,1,0, 0,1,0, 0,1,0],  # Vmiddle → 4
    [0,0,1, 0,0,1, 0,0,1],  # Vright  → 5
], dtype=float)

labels=[0,1,2,3,4,5]

line_names=["Htop", "Hmiddle", "Hdown", "Vleft", "Vmiddle", "Vright "]
