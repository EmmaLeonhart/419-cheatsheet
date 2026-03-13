# COSC 419/519 DEEP LEARNING - FORMULA SHEET & CRAM NOTES

---

## PAGE 1: LINEAR MODELS, LOSS FUNCTIONS, CLASSIFICATION

### Linear Regression
- Model: `y_hat = x^T * w` where x is (N+1)x1 (augmented with 1 for bias), w is (N+1)x1
- For full dataset: `Y_hat = X * w` where X is Px(N+1), w is (N+1)x1
- **MSE Loss**: `MSE = (1/P) * sum_i (y_hat_i - y_i)^2`
- **Normal Equation (Direct Solution)**: `w = (X^T X)^(-1) X^T Y`
- Code: `W = np.linalg.inv(X.T @ X) @ X.T @ Y`

### Logistic Regression (Binary Classification, C=2)
- **Sigmoid**: `sigma(t) = 1 / (1 + e^(-t))`
- Model: `f(w, x) = sigma(x^T w)` -> outputs probability of class 1
- Decision: `y_hat > 0.5 -> class 1`, `y_hat <= 0.5 -> class 0`
- Boundary: `x^T w = 0.5` (N=1), or hyperplane `w0 + w1*x1 + ... + wN*xN = 0`

### Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| **Step** | `step(a) = 1 if a>=0.5, 0 if a<0.5` | NOT differentiable, can't optimize |
| **MSE** | `(1/P) sum(y_hat - y)^2` | Regression; bad for classification (flat gradients, no probabilistic output) |
| **Binary Cross-Entropy** | `-1/P sum[y*log(f) + (1-y)*log(1-f)]` | Binary classification |
| **Multi-class Cross-Entropy** | `-1/P sum_p sum_c 1{y_p=c} * log(P(y_p=c\|x_p,w))` | Multi-class classification |
| **Multi-class Perceptron** | `-1/P sum[max_c(x^T w_c) - x^T w_{y_p}]` | Multi-class (not smooth) |
| **Softmax (Multi-class CE)** | `-1/P sum log(e^{f_{w_c}(x)} / sum_j e^{f_{w_j}(x)})` | Multi-class (smooth, convex) |

### Cross-Entropy Key Facts
- `loss_p = -log(probability of true class)` -- simplified form
- If predicted prob -> 0 for true class: loss -> infinity
- If predicted prob -> 1 for true class: loss -> 0
- CE penalizes violations much more harshly than MSE
- **If predicted=1 and actual=0: loss approaches INFINITY**

### Multi-Class Classification (C > 2)
- **One-vs-All**: Train C binary classifiers, fusion rule: `y_hat = argmax_c (x^T w_c)`
- **Softmax function**: `softmax(z_i) = e^{z_i} / sum_j e^{z_j}` -- outputs sum to 1
- Each z_i = x^T w_i (raw score/logit for class i)
- **One-Hot Encoding**: true label as vector [0,0,1,0,...] for class c
- **Softmax is generalization of sigmoid**: When C=2, softmax reduces to sigmoid
- Softmax invariant to constant shift: subtracting same vector psi from all w doesn't change predictions

### Logistic vs Softmax Summary

| Feature | Logistic | Softmax |
|---------|----------|---------|
| Classes | C=2 | C>=2 |
| Output | Scalar probability | C-dim vector |
| Activation | Sigmoid | Softmax |
| Parameters | 1 weight vector | C weight vectors (matrix) |
| Loss | Binary CE | Multi-class CE |
| Decision | P(y=1) > 0.5 | argmax_c P(y=c) |

---

## PAGE 2: OPTIMIZATION, REGULARIZATION, MLP & BACKPROP

### Gradient Descent
- **Update rule**: `W_{t+1} = W_t - alpha * grad_J(W_t)`
- **Gradient**: `grad_J = (1/N) sum_i grad_J_i(W)` (average over batch)
- alpha = learning rate; too small = slow, too large = diverge/oscillate

### Optimizer Comparison

| Optimizer | Pseudocode | Pros | Cons |
|-----------|-----------|------|------|
| **Full Batch GD** | `dW = grad(all_data); W -= lr*dW` | Most accurate gradient | Slowest, impractical for large data |
| **Mini-batch SGD** | `dW = grad(minibatch); W -= lr*dW` | Good speed/accuracy balance | Noisy updates, jitter |
| **SGD+Momentum** | `v = rho*v + dW; W -= lr*v` | Damps oscillation, escapes local min, builds speed | Extra hyperparameter rho |
| **AdaGrad** | `gs += dW*dW; W -= lr*dW/sqrt(gs+eps)` | Per-param adaptive lr, good for sparse | gs grows forever, lr decays to 0 |
| **RMSProp** | `gs = decay*gs + (1-decay)*dW*dW; W -= lr*dW/sqrt(gs+eps)` | Fixes AdaGrad's decay problem | Still no momentum |
| **Adam** | `m = b1*m+(1-b1)*dW; v = b2*v+(1-b2)*dW*dW; m_hat=m/(1-b1^t); v_hat=v/(1-b2^t); W -= lr*m_hat/sqrt(v_hat+eps)` | Best of both worlds, bias correction | Slightly more complex |

**Adam defaults**: beta1=0.9, beta2=0.999, lr=1e-3 or 5e-4

| Algorithm | 1st moment | 2nd moment | Leaky 2nd | Bias correction |
|-----------|:---:|:---:|:---:|:---:|
| SGD | - | - | - | - |
| SGD+Momentum | Y | - | - | - |
| AdaGrad | - | Y | - | - |
| RMSProp | - | Y | Y | - |
| Adam | Y | Y | Y | Y |

### Terminology
- **Epoch**: 1 complete pass through entire dataset
- **Batch size**: samples per gradient step. If data=200, batch=5 -> 40 batches/epoch
- **Learning rate**: step size for weight update

### Regularization

| Method | Formula/Description | Effect |
|--------|-------------------|--------|
| **L2** | `R(W) = sum W_ij^2` | Spreads weights evenly, prevents large weights, handles multicollinearity |
| **L1** | `R(W) = sum \|W_ij\|` | Drives weights to 0 -> feature selection, sparsity |
| **Elastic Net** | `lambda1*L1 + lambda2*L2` | Combines both benefits |
| **Dropout** | Random zero neurons (p=0.5 typical) | Forces redundant representations, ensemble effect |
| **Batch Norm** | Normalize activations per mini-batch | Stabilizes training, allows higher lr |
| **Data Augment** | Flip, crop, color jitter, cutout | More training variety |

- Total loss: `g(W) = (1/P) sum L(f(W,x_i), y_i) + lambda*R(W)`
- **L2 example**: w1=[1,0,0,0] vs w2=[.25,.25,.25,.25] -> L2 prefers w2 (0.25 < 1.0)
- **L1 example**: same vectors -> L1 equal penalty (both=1), but L1 prefers w1 (sparsity)
- **Dropout**: Train: drop p=0.5 of neurons. Test: multiply by p OR inverted dropout (scale during train, test unchanged)
- **Order**: Linear -> BatchNorm -> Activation -> Dropout

### MLP (Multi-Layer Perceptron)
- Feed-forward neural network with fully connected layers
- **Non-linearity needed**: Without it, stacking layers = single linear layer (can't solve XOR)
- **Forward pass**: h = f(W*x + b), layer by layer
- **Backpropagation**: Chain rule through computational graph
  - Compute loss, propagate gradients backward
  - Each node: local gradient * upstream gradient
- **Computational graphs**: Nodes = operations, edges = data flow
  - Explicit Jacobian: expensive (high memory, slow for deep nets)
  - Solution: Reverse-mode autodiff (backpropagation) -- implicit Jacobian

### Feature Transforms
- `y = w^T * f(x)` where f(x) is a non-linear feature map
- Polynomial: f(x) = (1, x, x^2, ..., x^M)
- Before DL: manual feature engineering (SIFT, HoG, Bag of Words)
- DL: features learned automatically

---

## PAGE 3: CNN, CNN ARCHITECTURES, TRAINING NETS

### CNN Fundamentals
- **CNN vs MLP**: CNN uses local receptive fields + weight sharing; MLP is fully connected
- **Weight sharing**: Same filter reused across all spatial positions -> massive param reduction
  - Example: 28x28 input, 100 neurons: FC=78,400 params vs CNN 32 3x3 filters=288 params

### CNN Components
1. **Conv Layer**: `output = input * filter` (sliding dot product)
   - Filter always extends full depth of input (e.g., 5x5x3 for RGB)
   - N filters -> N channels in activation map
2. **Activation** (ReLU): `max(0, x)`
3. **Pooling**: Downsample spatial dims (Max or Average)
   - Pooling has 0 learnable parameters!
4. **FC Layers**: Classification at end

### CNN Dimension Formula
```
Output = floor((W + 2P - F) / S) + 1
```
W = input size, F = filter size, P = padding, S = stride

**CONV weights**: `(F * F * C_in) * C_out`   (+ C_out biases)
**CONV output size**: `((W+2P-F)/S + 1) x ((H+2P-F)/S + 1) x C_out`
**Pooling output**: `(W-F)/S + 1` (no learnable params, depth unchanged)
**FC weights**: `input_flattened * neurons`

### Exam CNN Calculation (from past midterm)
INPUT 128x128x3:
- CONV-9-32: out=(128-9)/1+1=120 -> **120x120x32**, weights=9*9*3*32=**7,776**, biases=**32**
- POOL-2: out=120/2=60 -> **60x60x32**, weights=**0**, biases=**0**
- CONV-5-64: out=(60-5)/1+1=56 -> **56x56x64**, weights=5*5*32*64=**51,200**, biases=**64**
- POOL-2: out=56/2=28 -> **28x28x64**, weights=**0**, biases=**0**
- CONV-5-64: out=(28-5)/1+1=24 -> **24x24x64**, weights=5*5*64*64=**102,400**, biases=**64**
- POOL-2: out=24/2=12 -> **12x12x64**, weights=**0**, biases=**0**
- FC-128: flatten=12*12*64=9216 -> weights=9216*128=**1,179,648**, biases=**128**
- FC-3: weights=128*3=**384**, biases=**3**

### CNN Architecture Timeline

| Net | Year | Layers | Error | Key Innovation |
|-----|------|--------|-------|---------------|
| **AlexNet** | 2012 | 8 | 15.3% | First CNN winner; ReLU, dropout, data augment, GPU training |
| **ZFNet** | 2013 | 8 | 11.2% | Tweaked AlexNet (7x7 stride 2 instead of 11x11 stride 4) |
| **VGG** | 2014 | 16/19 | 7.3% | Only 3x3 convs; deeper is better; 3 stacked 3x3 = 7x7 receptive field with fewer params (27C^2 vs 49C^2) |
| **GoogLeNet** | 2014 | 22 | 6.67% | Inception module; 1x1 conv bottleneck to reduce computation; no FC layers; only 5M params |
| **ResNet** | 2015 | 152 | 3.6% | Skip/residual connections; solves vanishing gradient in very deep nets; H(x)=F(x)+x |
| **DenseNet** | 2016 | 121-265 | 5.3% | Dense connections (every layer connected to every other) |
| **ConvNeXt** | 2022 | - | - | Modernized CNN with transformer design principles; CNN comeback |
| **InternImage** | 2023 | - | - | Dynamic convolution kernels; captures global+local dependencies |

### Activation Functions

| Function | Formula | Pros | Cons |
|----------|---------|------|------|
| **Sigmoid** | 1/(1+e^-x) | Output [0,1] | Vanishing gradients, not zero-centered |
| **Tanh** | tanh(x) | Zero-centered [-1,1] | Still vanishing gradients |
| **ReLU** | max(0,x) | Fast, no saturation for x>0 | Dying ReLU (grad=0 when x<0) |
| **Leaky ReLU** | max(0.01x, x) | Won't die | Small negative slope |
| **GELU** | x*Phi(x) | Smooth near 0, good for transformers | Higher compute cost |

**Best practice**: Use ReLU (default), Leaky ReLU/GELU for marginal gains. Avoid sigmoid/tanh in hidden layers.

### Weight Initialization
- **All same value**: BAD - neurons learn same features (symmetry breaking fails)
- **Small random (std=0.01)**: OK for shallow, activations collapse in deep nets
- **Xavier/Glorot**: `std = 1/sqrt(D_in)` -- for sigmoid/tanh
- **Kaiming/He**: `std = sqrt(2/D_in)` -- for ReLU/Leaky ReLU
- Wrong init -> vanishing or exploding activations -> no learning

---

## PAGE 4: RNN, LSTM, DEBUGGING & EXAM TIPS

### Recurrent Neural Networks (RNN)
- Designed for sequential, variable-length data (NLP, time series)
- **Hidden state**: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
- **Output**: `y_hat_t = softmax(W_y * h_t + b_y)`
- Same weights W_hh, W_xh, W_y reused at EVERY time step
- **Dimensions**: x_t in R^d, W_xh in R^{D_h x d}, W_hh in R^{D_h x D_h}, W_y in R^{|V| x D_h}

### RNN Patterns
- **One-to-Many**: Image captioning (image -> sequence of words)
- **Many-to-One**: Sentiment analysis (sequence -> single label)
- **Many-to-Many**: POS tagging (aligned), Translation (Seq2Seq encoder-decoder)

### RNN Loss
- Per-timestep: `J_t = -sum_j y_{t,j} * log(y_hat_{t,j})` (cross-entropy)
- Total: `J = (1/T) sum_t J_t`

### Backprop Through Time (BPTT)
- `dJ/dW_hh = sum_t dJ_t/dW_hh`
- `dJ_t/dW_hh = sum_{k=1}^{t} (dJ_t/dh_t)(dh_t/dh_k)(dh_k/dW_hh)`
- Critical Jacobian: `dh_t/dh_k = prod_{j=k+1}^{t} [diag(f'(z_j)) * W_hh]`

### Vanishing & Exploding Gradients
- Repeated multiplication of W_hh: governed by largest eigenvalue lambda_max
- **lambda_max < 1**: gradients vanish exponentially (e.g., 0.5^50 ~ 10^-16)
  - Symptom: works on short sequences, fails on long; loses topic
- **lambda_max > 1**: gradients explode (e.g., 1.5^50 ~ 637M)
  - Symptom: NaN weights, loss -> infinity, training crashes
- **Fix exploding**: Gradient clipping: if ||g|| > threshold, scale g = g * threshold/||g||
- **Fix vanishing**: Need architectural change -> LSTM

### LSTM (Long Short-Term Memory)
- **Dual state**: Cell state C_t (long-term) + Hidden state h_t (short-term)
- Cell state = "conveyor belt" with only linear ops (add/multiply) -> easy gradient flow
- **4 Gates** (all use sigmoid to output [0,1]):
  1. **Forget gate (f)**: What to erase from cell state
  2. **Input gate (i)**: What new info to store
  3. **Cell gate (g)**: Candidate new values (tanh)
  4. **Output gate (o)**: What to output from cell state
- Key: backprop from c_t to c_{t-1} is only elementwise multiply by f (no matrix multiply!)

### Language Modeling
- Task: predict next word `P(x_{t+1} | x_1,...,x_t)`
- Joint probability: `P(x_1,...,x_T) = prod P(x_t | x_1,...,x_{t-1})`
- **Perplexity** = exp(J(theta)) -- lower = better
- **n-gram**: Markov assumption, P(w_n | w_{n-1},...,w_1) approximated by preceding n-1 words
- **Neural LM**: Fixed window -> asymmetry problem. RNN solves with shared weights

### Debugging Cheat Sheet (From Past Midterm)

| Problem | Cause | Fix |
|---------|-------|-----|
| **Dying ReLU** | Input always negative, gradient=0 | Leaky ReLU, better init, lower lr, batch norm |
| **Sigmoid vanishing** | Activations -> 0 or 1 in deep nets | Xavier init, batch norm, switch to ReLU |
| **Loss explodes to NaN** | Both lr too high AND/OR bad init | Reduce lr, use He/Xavier init |
| **Underfitting** | Model too simple | More layers, more neurons, reduce regularization |
| **Overfitting** | Model too complex | More data, dropout, L2, data augmentation |
| **All same weights init** | Symmetry problem | Neurons learn identical features; use random init |

### Learning Rate Curves (Exam Pattern)
- **Too high lr (e.g., 0.4)**: Loss oscillates wildly or explodes
- **Too low lr (e.g., 1e-5)**: Loss barely decreases, flat curve
- **Good lr (e.g., 3e-4)**: Smooth decrease to low loss

### Full Batch GD vs Adam Curves
- **Full Batch GD**: Smooth, monotonically decreasing curve
- **Adam**: Faster initial decrease, may oscillate slightly but converges faster

### Key Exam MCQ Answers (From Past Paper)
1. Dropout: only during training, probability is hyperparameter, applied differently train/test
2. Activation function role: introduce non-linearity
3. 1x1 conv in Inception: reduce dimensionality
4. InternImage: dynamic conv kernels, global+local dependencies
5. Underfitting fix: more layers, reduce regularization (NOT more dropout)
6. Same weight init: neurons learn same features
7. CE loss: pred=1, actual=0 -> loss approaches infinity
8. MSE bad for classification: doesn't penalize effectively, no probabilistic output
9. OvA: train C classifiers + fusion rule, learn all weights simultaneously
10. Softmax+CE requirement: outputs must sum to 1
11. SGD vs BGD: SGD updates per example; decaying lr improves stability
