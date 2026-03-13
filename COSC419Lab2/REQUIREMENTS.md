# COSC 419/519B - Deep Learning - A2 (Lab 2)

**Course:** COSC 419/519B - Deep Learning, Winter 2025 T2
**Instructor:** Dr. M. S. Shehata
**Due:** Feb. 9th, 2026
**Total Points:** 20
**Focus:** PyTorch, ANN, Optimization
**Submission:** Jupyter notebook to Canvas (must run without errors, no group work)

---

## Section 1: Setting Up PyTorch and Data [4 points]

### Task 1.1 - Import PyTorch
- Import `torch` and `matplotlib`
- Set manual seed to 42 for reproducibility

### Task 1.2 - Create Synthetic Data
- Generate 200 data points, 2 classes, moon-shaped distribution
- Use `sklearn.datasets.make_moons`

### Task 1.3 - Visualize the Data
- Scatter plot with matplotlib
- Class 0 = blue, Class 1 = orange
- Axes: "Feature 1", "Feature 2"
- Title: "Visualization of the moons Dataset."

### Task 1.4 - Split the Dataset
- 80% train / 20% test via **slicing** (not `train_test_split`)
- X -> `torch.FloatTensor`, y -> `torch.LongTensor`
- Print shapes of all splits

---

## Section 2: Building a Neural Network [2 points]

### Task 2.1 - Define SimpleNN
- Class inheriting `nn.Module`
- Layer 1: `nn.Linear(2, 8)`
- Activation: `nn.ReLU()`
- Layer 2: `nn.Linear(8, 2)`
- Instantiate the model

### Task 2.2 - Forward Pass
- Pass a single training data point through the model
- Print output
- **Answer:** What do these outputs represent? (raw logits / unnormalized class scores)

---

## Section 3: Training the Neural Network [3 points]

### Task 3.1 - Loss Function and Optimizer
- Loss: `nn.CrossEntropyLoss`
- Optimizer: `optim.Adam`, lr=0.01

### Task 3.2 - Training Loop
- 200 epochs
- Each epoch: zero_grad -> forward -> loss -> backward -> step
- Print loss every 20 epochs
- Store losses in array `losses`
- Plot loss curve after training

### Task 3.3 - Evaluation
- Predict on test set using `torch.argmax()`
- Count correct predictions
- Calculate and print accuracy

---

## Section 4: Deep Dive into PyTorch Optimizers [3 points]

### Task 4.1 - Theory Questions
Answer these about `torch.optim.Optimizer`:
- **(a)** Purpose of `params` in `__init__` and how optimizer tracks model parameters
- **(b)** Structure of `param_groups` - what is a param group, why list of dicts?
- **(c)** What `state` stores and how optimizer uses it
- **(d)** Why `zero_grad()` is essential before each iteration (gradients accumulate otherwise)
- **(e)** Role of `step()` in abstract terms

### Task 4.2 - Implement DummyOptimizer
- Inherits `torch.optim.Optimizer`
- `__init__`: takes `params`, `lr`, stores in `defaults`, calls `super()`
- `step`: iterates `param_groups` -> params -> updates via `p.data.add_(-group['lr'], p.grad.data)`
- Essentially simplified SGD

### Task 4.3 - Test DummyOptimizer
- New `SimpleNN` instance
- Use `DummyOptimizer` in training loop
- 50 epochs
- Print loss periodically to verify learning

---

## Section 5: Implementing AdamW with Decoupled Weight Decay [10 points]

### Task 5.1 - Theory Questions
- **(a)** Difference between weight decay and L2 regularization; why not equivalent in Adam
- **(b)** How AdamW decouples weight decay from gradient update; key difference vs Adam
- **(c)** Why decoupling improves generalization (per the paper)

**Reference paper:** "Decoupled Weight Decay Regularization" - Loshchilov & Hutter (2019)
https://arxiv.org/abs/1711.05101

### Task 5.2 - Implement AdamW
- Inherits `torch.optim.Optimizer`
- `__init__`: lr, betas, eps, weight_decay -> stored in `defaults`; init state vars (exp_avg, exp_avg_sq)
- `step`: follows AdamW pseudocode from paper
  - Key line: `p.data.mul_(1 - group['lr'] * group['weight_decay'])` (decoupled weight decay)
- Use `@torch.no_grad()` decorator

**Pseudocode from assignment:**
```
1. Init: theta_0, lr schedule {alpha_t}, beta_1, beta_2, eps, weight_decay lambda
2. Init moments: m_0 = 0, v_0 = 0
3. for t = 1..T:
   4. g_t = gradient
   5. m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
   6. v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
   7. m_hat = m_t / (1 - beta_1^t)
   8. v_hat = v_t / (1 - beta_2^t)
   9. theta = theta - alpha_t * m_hat / (sqrt(v_hat) + eps)
   10. theta = theta - alpha_t * lambda * theta_{t-1}   # decoupled weight decay
```

### Task 5.3 - Learning Rate Scheduler
- Use `torch.optim.lr_scheduler.StepLR`
- Reduce lr by factor 0.1 every 50 epochs

### Task 5.4 - Testing and Comparing Optimizers
1. Train with **custom AdamW** + LR scheduler
2. Train with **torch.optim.Adam** + LR scheduler
3. Train with **torch.optim.AdamW** + LR scheduler
4. Same initial lr across all
5. **Plot** all 3 loss curves on one graph
6. Print final training loss + test accuracy for each
7. Experiment with weight_decay values: 0, 0.01, 0.1
8. **Analysis questions:**
   - **(a)** How does custom AdamW compare to Adam and torch.optim.AdamW?
   - **(b)** Effect of different weight_decay values on training and generalization?

---

## Notebook Cell Map

| Cell | Task | Type |
|------|------|------|
| cell-2 | 1.1 Import PyTorch | Code |
| cell-4 | 1.2 Create Synthetic Data | Code |
| cell-6 | 1.3 Visualize Data | Code |
| cell-8 | 1.4 Split Dataset | Code |
| cell-10 | 2.1 Define SimpleNN | Code |
| cell-12 | 2.2 Forward Pass | Code |
| cell-14 | 3.1 Loss + Optimizer | Code |
| cell-16 | 3.2 Training Loop | Code |
| cell-18 | 3.3 Evaluation | Code |
| (new) | 4.1 Theory Questions | Markdown (insert after cell-20) |
| cell-22 | 4.2 DummyOptimizer | Code |
| cell-24 | 4.3 Test DummyOptimizer | Code |
| (new) | 5.1 Theory Questions | Markdown (insert after cell-26) |
| cell-28 | 5.2 Implement AdamW | Code |
| cell-30 | 5.3 LR Scheduler | Code |
| cell-32 | 5.4 Compare Optimizers | Code |
