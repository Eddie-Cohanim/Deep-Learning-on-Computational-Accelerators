r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Answer 1:**$\\$
The shape of the resulting tensor is ${R}^{64x512x64x1024}$.

**Answer 2:**$\\$
Since the Jacobian matrix will be a block matrix of NxN, where each
block is 1024x512, we can index the samples using n,m such that the
block(n,m) is the jacobian matrix of the output of sample n over the
input of sample m, $\frac{\partial Y^{(n)}}{\partial X^{(m)}}$. and since the linear layers act independently
per sample we get:
$${Y}^{n}=W*{X}^{n}$$

now we shall look at 2 differenct cases:$\\$
$\hspace{2em}$ **i**. $n \neq m$ : 
$$\frac{\partial Y^{(n)}}{\partial X^{(m)}} = 0$$
$\hspace{4em}$ Which means there is no dependence between samples.$\\$

$\hspace{2em}$ **ii**. $n = m$ :
$$\frac{\partial Y^{(n)}}{\partial X^{(m)}} = W$$
$\hspace{4em}$ Which means that we get the same linear map for every
sample.$\\$

Therefore, The block matrix is block diagonal with N identical
diagonal blocks equal to W and all off diagonal blocks equal zero.

**Answer 3:**$\\$
Since the Jacobian matrix is constructed from blocks on the diagonal
that are W, with all other blocks equaling 0, there is no need to
construct the whole matrix, instead we can just construct the non zero
elemnts in the matrix, which is the blocks on the diagonal, and since
they are all W we only need W once, which is the shape of
${R}^{512x1024}$

**Answer 4:**$\\$
beacuse the Jacobian matrix is block diagonal with W equaling the blocks,
we can calculate the downstream gradient using the chain rule.$\\$
For each sample we get:
$$y = Wx $$ $$\frac{\partial Y}{\partial X} = W$$ 

Therefor we get:
$$\frac{\partial L}{\partial x} = (\frac{\partial y}{\partial x})^{\top} \frac{\partial L}{\partial y}
 = W^{\top}\delta y$$
Which leads to:
$$\delta X = \delta YW$$

**Answer 5:**$\\$
When differntiating, we get that 
$\frac{\partial {Y}_{n,m}}{\partial {W}_{k,i}}$ equals either ${X}_{n,i}$ when
$k = m$ **or** $0$ otherwise$\\$
Therefore, the JAcobian matrix in non zero only along the output dimension diagonal and its
shape is: 
$$R^{64x512x512x1024}$$
with each$(m,k)$ slice forming an $(512x1024)$ block.
"""

part1_q2 = r"""
**Answer:**$\\$
the second order derivative can be useful, such as when the loss landscape is curved very
differently along different directions(flat in one direction but very steep in another) will
cause the gradient descent to take small, slow steps. using the second order derivative will
allow the optimizer to take large steps in flat directions and small steps in steeper ones,
leading to faster convergance.


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.05
    reg = 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.02
    lr_momentum = 0.004
    lr_rmsprop = 0.0001
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Answer:**$\\$
Yes, it is possible for test loss to decrease while test accuracy decreases when using 
cross-entropy.
A model can assign higher probability to the correct class without crossing the decision
boundary needed to change the predicted label. In that case the loss goes down because
the probabilities improved, but accuracy stays the same or even gets worse if other 
samples flip from correct to incorrect.
"""

part2_q3 = r"""
**Answer 3: **$\\$

1. The suggested method does not produce the same gradient as full batch gradient descent.
In gradient descent the gradient is computed across the entire dataset and averaged. 
The proposed approach instead computes the loss for each batch, sums those losses, and 
then backpropagates once through the total. This means the gradient is based on the sum 
of batch losses rather than the mean across all samples, which scales the gradient and 
changes the update step.$\\$

$\hspace*{2em}$ Standard GD computes:
$$\nabla L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla l(\theta; x_i, y_i)$$

$\hspace*{2em}$ while the suggested approach computes:
$$\nabla L_{\text{total}}(\theta) = \sum_{j=1}^{K} \sum_{(x_i, y_i) \in B_j} \nabla l(\theta; x_i, y_i)$$

$\hspace*{2em}$ with N samples, K batches, and ${B}_{j}$ as batch j. The difference is that the gradient
is summed instead of averaged, so the update step is larger.$\\$
$\hspace*{2em}$ However, if the gradients are averaged across all batches by dividing by N, the result becomes 
equivalent to standard GD.$\\$

2. The out of memory issue likely came from memory resources accumulating across batches.
If intermediate values such as activations or gradients are not released after each 
batch, they remain stored and build up, eventually exceeding available memory. 
During forward and backward passes, training produces many temporary tensors that 
increase memory consumption. This effect can be reduced by releasing intermediate values 
as soon as they are no longer required, ensuring memory is cleared between batches, 
or applying strategies like gradient accumulation to keep usage manageable.$\\$


3. A possible solution woulb be to backpropagate after each batch so the graph is freed
immediately, or use gradient accumulation where each batch contributes a scaled loss
and "loss.backward()" is called repeatedly without storing previous graphs. This 
prevents memory growth over batches and removes the out of memory problem.$\\$
"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Answer:**
When optimizing for a lower FPR, we aim to reduce false positives even if this 
increases false negatives. This may be desirable in a system where incorrectly 
labeling a negative instance as positive carries a higher cost than missing some 
true positives. In such a case we accept a higher FNR because optimizing FPR is 
more important.

When optimizing for a lower FNR, we want to avoid false negatives even if this 
increases false positives. This match scenarios where failing to find a true 
positive is more damaging than mistakenly classifying some negatives as positive. 
Here we accept a higher FPR because optimizing FNR is the priority.

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Answer 1:**$\\$
The model shows two detections labeled person with confidences 0.50 and 0.90, 
and a third detection labeled surfboard with 0.37 confidence. None of the 
detections match the actual objects. The bounding boxes surround dolphins 
jumping from the water, so the model incorrectly interprets their shape and 
posture as people and mistakes part of the tail of the dolphin on the right as a 
surfboard. The only confident prediction is the second person box (0.90) 
though it is still wrong. Overall performance on this image is very bad.$\\$

in the second image the model detects 2 cats with confidences 0.66 and 0.39, 
and labels one dog as dog with a confidence of 0.51. In reality the image 
contains three dogs and one cat. It correctly identifies the dog in the center, 
but mislabels two dogs as cats and does not detect the remaining cat at all. 
Detection quality is better here than in the first image, yet multiple errors 
still occur. Overall detection is usuccessful and inconsistent. $\\$

**Answer 2:**$\\$
There are many reasons why the model might fair, for example:$\\$

1. YOLOv5 is trained on COCO, which does not include a dolphin class, so it 
must map dolphins to the closest existing label, often person. $\\$
Solution: retrain the model with a dataset that includes dolphins, 
dogs, and cats across diverse conditions.$\\$

2. If the calsses share the same or similar features, such as dogs and cats both
have fur, ears, four legs etc.$\\$
Solution: train on more diverse datasets with harder examples
and clearer inter-class boundaries.$\\$

3. Insufficient training data for rare or visually similar classes.$\\$
Solution: data augmentation, class-balanced sampling, increasing representation
of confusing pairs in the training set.$\\$

4. weak feature representation and poor calibration may lead to high confidence
in wrong predictions.$\\$
Solution: fixes include using calibration techniques, further 
training, adjusting the detection layers, or increasing the strength and 
capacity of the feature extraction.$\\$

**Answer 3:**$\\$
To attack YOLO with PGD, we introduce small, iterative perturbations to the 
input image that increase the model's loss while remaining visually subtle. 
PGD updates the image using the gradient of the detection loss, gradually 
pushing the model toward errors. Since YOLO predicts both class labels and 
bounding boxes, the attack can force misclassification, shift bounding boxes, 
or cause missed detections.

"""


part6_q2 = r"""
**Your answer:**

question number 2 was removed from the assignment by course staff.

"""


part6_q3 = r"""
**Your answer:**$\\$
image 1 "cat hidding in sofa - occlusion":$\\$
The cat is correctly detected and classified with 0.77 confidence, and there 
arn't any additional detections in the background. This is a case where the 
model performs as expected, with accurate classification and localization,
even though there was a mild case of occlusion in the image(since half of
the cat is obstructed by the sofa).$\\$

imgae 2 "street filled with cars and motorcycles - cluttered background":$\\$
The model recognized many cars, some trucks, and a number of people, 
although many objects remained undetected. the image contains a dense, highly 
cluttered enviroment with overlapping objects, which leads to a hard time making
accurate detections, therby leading to missed bounding boxes and low confidence 
scores in several detections. This image demonstrates how crowding can reduce 
detection coverage.$\\$

image 3 "dog running - motion blur":$\\$
The model successfully identified the dog with 0.83 confidence, which is high.
This is a good result given the strong motion blur and distortion in the image. 
However, it also incorrectly labeled part of the background as a surfboard with 
0.55 confidence. we can infer this confusion was caused by streaks and elongated
edges created by the motion blur. Overall, object detection is reasonably 
accurate for the dog, but noise and blur lead to an additional false detection.
$\\$

"""

part6_bonus = r"""
**Answer:**$\\$
Motion Blur Image:$\\$
We applied gentle sharpening and additional contrast to the image. The dog was still
correctly detected with 0.82 confidence, but now no surfboard was detected in 
the background.$\\$

Cluttered Background Image:$\\$
We split the image horizontally and left the top half unchanged, while appliing 
additional contrast to the bottom half only. This resulted in additional correct
people detections and an elimination of a false bus detection on the left that
overlapped with a truck detection.$\\$
"""