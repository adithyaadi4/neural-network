# 🧠 Neural Network from Scratch – Line Detection

A simple project built to understand neural networks.  
This project implements a neural network from scratch to detect line patterns in a 3×3 grid.

---

## 🎯 Patterns Learned

### Horizontal Lines

```
Htop
1 1 1
0 0 0
0 0 0

Hmiddle
0 0 0
1 1 1
0 0 0

Hdown
0 0 0
0 0 0
1 1 1
```

---

### Vertical Lines

```
Vleft
1 0 0
1 0 0
1 0 0

Vmiddle
0 1 0
0 1 0
0 1 0

Vright
0 0 1
0 0 1
0 0 1
```

---

## 🚀 What It Does

- Detects all the above line patterns  
- Learns from training data  
- Improves predictions over time  

---

## ⚙️ How It Works

- Takes a 3×3 grid as input  
- Processes it through a neural network  
- Uses:
  - ReLU activation  
  - Softmax output  

---

## 🔁 Training

- Trained over multiple epochs  
- Uses backpropagation to update weights  
- Gradually learns correct classifications  

---

## 💡 Example Output

```
Input:
1 1 1
0 0 0
0 0 0

Prediction → Htop ✅
```

---

## 📌 Note

> Built using pure NumPy — no ML frameworks used.
