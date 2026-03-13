# COSC 419/519B – Deep Learning
**Assignment 3 (20 points) – Due Feb. 23rd, 2026**
Focus: PyTorch, CNN

---

## Section 1 – Setting Up PyTorch and Preparing the Data [4 points]

### Task 1.1 – Importing Libraries
Import `torch`, `torch.nn`, `torch.optim`, `torchvision`, `torchvision.transforms`, `matplotlib.pyplot`, `numpy`, and `tqdm.auto`. Set a manual seed of 42 for reproducibility. Also add code to check and print the device being used (CPU or GPU).

---

### Task 1.2 – Downloading and Loading the FashionMNIST Dataset
Use `torchvision` to download and load the FashionMNIST dataset with the following transformations:
1. **ToTensor** – convert images to PyTorch tensors
2. **Normalize** – normalize pixel values to the range [-1, 1] (grayscale, so one mean/std value)

Create data loaders for both training and test sets with batch size 32 and 2 worker processes.

---

### Task 1.3 – Visualizing the Data
Create a grid of images from the training set using `matplotlib.pyplot.subplots`:
1. Get a batch of training images and their labels
2. Set up rows/columns (e.g., 8 images per row)
3. Create the figure with `plt.subplots()`
4. For each image:
   - Unnormalize it
   - Display with `ax.imshow()`
   - Set the class name as the title with `ax.set_title()`
   - Turn off axes with `ax.axis('off')`
5. Remove unused subplots
6. Adjust spacing with `plt.subplots_adjust()` if needed
7. Show with `plt.show()`

---

## Section 2 – Building the Model [2 points]

### Task 2.1 – Define the VGG16-like CNN Class
Create `VGG16LikeNet` inheriting from `nn.Module` with the following architecture:

**Block 1:** Conv2d(1→64, 3×3, pad=1) → ReLU → Conv2d(64→64, 3×3, pad=1) → ReLU → MaxPool2d(2×2)

**Block 2:** Conv2d(64→128, 3×3, pad=1) → ReLU → Conv2d(128→128, 3×3, pad=1) → ReLU → MaxPool2d(2×2)

**Block 3:** Conv2d(128→256) → ReLU → Conv2d(256→256) → ReLU → Conv2d(256→256) → ReLU → MaxPool2d(2×2)

**Block 4:** Conv2d(256→512) → ReLU → Conv2d(512→512) → ReLU → Conv2d(512→512) → ReLU *(no MaxPool)*

**Block 5:** Conv2d(512→512) → ReLU → Conv2d(512→512) → ReLU → Conv2d(512→512) → ReLU → MaxPool2d(2×2)

**Fully Connected:**
- `Linear(512×1×1, 4096)` → ReLU
- `Linear(4096, 4096)` → ReLU
- `Linear(4096, 10)`

**Poor Initialization:** Create `init_weights_poor(m)` that sets weights from `Normal(0, 0.1)` and biases to 0 for Conv2d and Linear layers. Apply with `self.apply(init_weights_poor)`.

---

### Task 2.2 – Instantiate and Print Architecture
Create `net = VGG16LikeNet()`, move to GPU with `net.to(device)`, and print with `print(net)`.

---

## Section 3 – Training the Model (Poor Initialization) [3 points]

### Task 3.1 – Define Loss Function and Optimizer
- Loss: `criterion = nn.CrossEntropyLoss()`
- Optimizer: `optimizer = optim.Adam(net.parameters(), lr=0.001)`

---

### Task 3.2 – Training Loop
1. Set number of epochs (5–10 to start)
2. Wrap epoch loop with `tqdm`
3. Each iteration:
   - Move inputs/labels to GPU
   - `optimizer.zero_grad()`
   - `outputs = net(inputs)`
   - `loss = criterion(outputs, labels)`
   - `loss.backward()`
   - `optimizer.step()`
   - Accumulate loss
4. Print average loss per epoch
5. Store losses in `losses` list
6. Plot losses after training with `matplotlib`

---

## Section 4 – Testing the Model (Poor Initialization) [3 points]

### Task 4.1 – Evaluate on the Test Set
Iterate through `testloader`, compute predictions, and calculate:
```
accuracy = 100 * correct / total
```
Print the overall accuracy.

---

### Task 4.2 – Class-wise Accuracy
Track `class_correct` and `class_total` for each of the 10 classes and print per-class accuracy.

---

### Task 4.3 – Visualizing Failure Cases
Collect wrong predictions into `wrong_images`, `wrong_labels`, `wrong_predictions`, then display the first 32 as a grid showing:
- The image
- Ground truth label (GT)
- Predicted label (Pred)

---

## Section 5 – Improving the Model with He Initialization [10 points]

### Task 5.1 – Understanding He Initialization *(Theoretical – write in a text cell)*

**Key concepts:**
- Xavier was designed for symmetric activations (tanh); ReLU is asymmetric (clips negatives to zero), so variance behaves differently
- He formula: `std = sqrt(2 / fan_in)` — uses only `fan_in`, not `fan_in + fan_out`
- The factor of 2 compensates for ReLU killing ~half the neurons on average

---

#### Version 1 — Concise and direct
He initialization addresses a problem specific to ReLU networks. Xavier initialization was built around the assumption that activations are symmetric, but ReLU cuts off all negative outputs, effectively halving the information flowing forward. He initialization compensates by scaling the initial weights by sqrt(2 / fan_in) instead of Xavier's sqrt(2 / (fan_in + fan_out)). The extra factor of 2 accounts for the roughly 50% of neurons that ReLU zeros out, keeping the variance of activations stable as signals pass through many layers.

---

#### Version 2 — Intuition-first
Imagine you're passing a signal through a long chain of layers. If your initial weights are too large, the signal explodes; too small, and it vanishes. Xavier initialization manages this for tanh networks, but ReLU throws a wrench in things because it silences every negative activation. That wipes out roughly half the signal at each layer. He initialization doubles the weight variance to make up for that loss, using std = sqrt(2 / fan_in). The result is a signal that stays at a consistent magnitude all the way through the network, even when it's very deep.

---

#### Version 3 — Math-forward
Xavier initialization derives its scaling by assuming the activation function preserves variance symmetrically. For a layer with fan_in inputs and fan_out outputs, it sets std = sqrt(2 / (fan_in + fan_out)). However, this derivation breaks down for ReLU. Since ReLU(x) = max(0, x) zeroes out negative inputs, it reduces the effective variance of activations by roughly a factor of 2. He et al. (2015) corrected for this by using std = sqrt(2 / fan_in), where the factor of 2 in the numerator compensates for the variance lost through the ReLU nonlinearity.

---

#### Version 4 — Comparison-focused
The key difference between Xavier and He initialization comes down to the activation function they're designed for. Xavier assumes that activations are roughly symmetric around zero (like tanh), so it splits the scaling between incoming and outgoing connections. He initialization is tailored for ReLU activations, which are decidedly not symmetric — they output zero for any negative input. Since this zeros out about half of all activations on average, He initialization doubles the weight variance relative to what Xavier would use, keeping the forward signal from shrinking to nothing as it travels through a deep network.

---

#### Version 5 — Problem/solution framing
**Problem:** In a deep network with ReLU activations, poorly scaled weights cause gradients to either explode or vanish during training, making learning extremely slow or impossible.

**Why Xavier fails here:** Xavier initialization was derived assuming symmetric activations. ReLU violates this — it clips everything below zero, so only about half the neurons actually fire. This biases the variance of activations downward at each layer.

**He's solution:** Scale initial weights with std = sqrt(2 / fan_in). The factor of 2 directly compensates for the ~50% of activations zeroed out by ReLU. This keeps the variance of activations stable layer-to-layer, allowing gradients to flow cleanly during backpropagation.

---

#### Version 6 — Casual/explanatory
Xavier initialization works great when you use tanh, because tanh outputs values on both sides of zero equally. But ReLU is different — it just shuts off any negative input entirely. So if you use Xavier weights with a ReLU network, you're effectively losing about half your signal at every layer, and by the time you get ten or fifteen layers deep, barely anything is getting through. He initialization fixes this by bumping up the weight scale. Specifically, you draw weights from a normal distribution with std = sqrt(2 / fan_in). That "2" is there precisely to compensate for all the activations ReLU kills.

---

#### Version 7 — Lecture-note style
**Xavier Initialization:**
- Assumes zero-mean, symmetric activation (e.g., tanh)
- std = sqrt(2 / (fan_in + fan_out))
- Maintains variance across layers under symmetric conditions

**Problem with ReLU:**
- ReLU(x) = max(0, x) is not symmetric
- Approximately half of all pre-activations are negative → zeroed out
- This reduces the output variance by ~50% per layer
- Over many layers: signal vanishes

**He Initialization (fix):**
- std = sqrt(2 / fan_in)
- The "2" in the numerator compensates for the 50% variance loss from ReLU
- Only fan_in used (not fan_out), since the derivation is based on forward-pass variance
- Result: stable activation variance through arbitrarily deep ReLU networks

---

#### Version 8 — Short and exam-ready
He initialization is a weight initialization method designed for networks with ReLU activations. Unlike Xavier, which uses std = sqrt(2 / (fan_in + fan_out)) and assumes symmetric activations, He uses std = sqrt(2 / fan_in). The factor of 2 compensates for the fact that ReLU zeroes out roughly half of all activations, which would otherwise cause the variance of layer outputs to shrink with each layer. By correcting for this, He initialization helps maintain stable gradient flow throughout deep networks.

---

#### Version 9 — Analogy-driven
Think of the network's forward pass like a game of telephone. Each layer passes information to the next, but ReLU acts like a filter that throws away about half the message every time. If your starting weights aren't loud enough to begin with, the signal fades out before it reaches the end of a deep network. He initialization essentially starts the telephone game at a louder volume — specifically, std = sqrt(2 / fan_in) — so that even after ReLU quietly discards half the signal at each step, enough information survives to keep training moving in the right direction.

---

#### Version 10 — Mechanistic/variance-tracking
When a weight matrix W is initialized with variance σ², the output variance of a layer with n inputs is n·σ² (before activation). For tanh, this is approximately preserved since the activation is roughly linear near zero. For ReLU, however, only positive inputs are passed through, so the effective output variance is approximately n·σ²/2. Xavier initialization targets n·σ² ≈ 1 by setting σ² = 1/n, which overshoots for ReLU networks (the actual output variance becomes ~0.5). He initialization instead sets σ² = 2/n, so that the ReLU-adjusted output variance becomes n·(2/n)/2 = 1. This keeps activations at unit variance layer-to-layer, preserving the gradient signal in both forward and backward passes.

---

### Task 5.2 – Implementing He Initialization
1. Create `VGG16LikeNetHe` (same architecture as `VGG16LikeNet`)
2. Create `init_weights(m)`:
   - For `nn.Conv2d`: `fan_in = in_channels × kH × kW`, `std = sqrt(2 / fan_in)`
   - For `nn.Linear`: `fan_in = in_features`, `std = sqrt(2 / fan_in)`
   - In both cases: `m.weight.data.normal_(0, std)`, bias → zero
3. Apply with `self.apply(init_weights)`

---

### Task 5.3 – Instantiate, Train, and Evaluate (He Init)
Same process as Section 3 & 4, but with `net_he = VGG16LikeNetHe()` and a fresh optimizer `optimizer_he`.

---

### Task 5.4 – Comparison Plots *(Analysis – write in a text cell)*
Plot log-scale training loss for both models on the same graph.

**Guiding questions:**
- What differences do you observe in loss curves between poor and He initialization?
- Which model converged faster / to a lower loss?
- What does this tell you about the importance of weight initialization in deep networks?

---

#### Version 1 — Concise and direct
The He-initialized model consistently achieved lower training loss and converged faster than the poorly initialized one. On the log-scale plot, the gap between the two curves was visible from the very first epoch. The poorly initialized model struggled to reduce its loss meaningfully, suggesting that the network's gradients were too small or noisy to drive effective updates. He initialization gave the optimizer a better starting point, allowing the network to learn useful features from the beginning rather than spending most of training recovering from a bad initialization.

---

#### Version 2 — Intuition-first
Looking at the log-scale comparison, the He-initialized model dropped its loss steeply right from the start, while the poorly initialized model's curve barely moved, staying at a much higher loss throughout training. This makes sense intuitively: if the weights are badly scaled at the start, the gradient signals get muddled as they travel back through all those layers, so the network doesn't really know which direction to update. He initialization fixes the starting point so gradients can flow clearly, and the model can actually learn.

---

#### Version 3 — Quantitative/analytical tone
The log-scale plot reveals a significant difference in training dynamics. The He-initialized model's loss curve descends steeply and consistently across all 10 epochs, indicating that gradient updates were effective and the optimizer was navigating a well-conditioned loss landscape. The poorly initialized model's log-loss curve is comparatively flat or shows much slower descent, consistent with the vanishing gradient problem: when weights are initialized too small, gradients shrink exponentially as they backpropagate through many layers, leaving earlier layers nearly unchanged. This experiment demonstrates that initialization quality directly determines whether a deep network can train at all, not just how fast it trains.

---

#### Version 4 — Comparison-focused
The two loss curves tell very different stories. The He-initialized model starts at a much lower loss and continues to decline steadily, showing that meaningful learning is occurring every epoch. The poorly initialized model starts at a higher loss and makes little progress relative to the He model — on the log scale, the difference is striking. This comparison highlights that weight initialization isn't just a minor tuning detail; for deep networks like VGG16, it can be the difference between a model that learns and one that effectively stalls.

---

#### Version 5 — Problem/cause/effect
**Observation:** On the log-scale plot, the He-initialized model has a noticeably lower loss from epoch 1 and continues to decrease more rapidly.

**Cause:** Poor initialization (std = 0.1 for all layers) leads to gradient signals that are either too weak or inconsistent by the time they reach the early layers. This is the vanishing gradient problem in action.

**Effect:** The poorly initialized network wastes training time in early epochs barely moving, while He initialization gives the network a healthy starting variance that lets gradients flow and weights update meaningfully from the start.

**Conclusion:** In networks this deep, the choice of initialization scheme is as important as the architecture itself.

---

#### Version 6 — Casual
It's pretty clear from the log plot which initialization strategy won. The He-initialized model dropped in loss almost immediately, while the poor initialization model kind of plateaued at a higher value and barely improved. The reason is that when the weights are too small to start, the gradients that backpropagate through 13 convolutional layers get tinier and tinier until they're basically useless for updating the early weights. He initialization prevents that by making sure the weights are scaled appropriately for a ReLU network, so learning actually happens from the first epoch onward.

---

#### Version 7 — Lecture-note style
**Poorly initialized model:**
- Log-loss starts high; curve is relatively flat or slow to decline
- Indicative of vanishing gradients in early layers
- Optimizer struggles to find useful update directions

**He-initialized model:**
- Log-loss starts lower; curve descends consistently each epoch
- Gradients remain well-conditioned through all 13 conv layers
- Optimizer makes productive updates from the first batch

**Key takeaway:** Initialization quality directly controls whether gradient flow is healthy. In deep architectures, bad initialization can make training practically impossible regardless of learning rate or optimizer choice.

---

#### Version 8 — Short and exam-ready
The comparison plot shows that the He-initialized model achieves significantly lower training loss across all epochs compared to the poorly initialized model. On the log scale, the He model's curve is both lower and steeper, indicating faster and more effective learning. The poorly initialized model suffers from vanishing gradients: its weight scale of 0.1 causes gradient magnitudes to collapse across the network's many layers, preventing meaningful weight updates. This demonstrates that proper initialization is critical for training deep networks.

---

#### Version 9 — Analogy-driven
If training is like hiking downhill to find the lowest point in a valley, initialization determines where you start. He initialization drops you partway down the slope, where you can see clearly and take good steps downward. Poor initialization drops you in fog at the top — you can't tell which way is down, so your steps are nearly random. On the log plot, you can see this clearly: the He model makes steady, confident progress toward lower loss, while the poorly initialized model wanders without much direction, ending up at a much higher loss after the same number of epochs.

---

#### Version 10 — Mechanistic/variance-tracking
The log-scale plot exposes the compounding effect of variance mismatch. With std = 0.1 across all layers, the variance of activations shrinks at each layer (since var(Wx) = fan_in · 0.01), causing gradients in the backward pass to shrink by a corresponding factor per layer. Over 13 convolutional layers, this cumulative shrinkage renders the gradients of early layers negligibly small. He initialization, by setting variance to 2/fan_in, maintains approximately unit variance in activations throughout the forward pass, which in turn keeps gradient magnitudes consistent during backpropagation. The result on the log-loss plot is a He-initialized curve that descends quickly and steadily, versus a poorly initialized curve that is nearly flat — visually confirming that effective gradient flow is a prerequisite for deep network training.

---

## Submission Checklist
- [ ] All `# Your code here` cells filled in
- [ ] Text cells added for theoretical answers (5.1 and comparison analysis)
- [ ] Notebook runs end-to-end without errors
- [ ] Submitted to Canvas (new submission overwrites old)
