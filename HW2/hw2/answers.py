r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
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
    raise NotImplementedError()
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
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Answer 1:**$\\$
Yes, the results clearly match what we expected to see. When the dropout rate is 0, the model strongly overfits the training data.
This is shown by the very low training loss and very high training accuracy, which approaches almost 100%. At the same time, the
test loss remains high and even increases slightly, while the test accuracy stays low. This large gap between training and test
performance indicates that the model is memorizing the training data instead of learning general features.$\\$

When dropout is added, the behavior changes in the expected way. With both dropout = 0.4 and dropout = 0.8, the training accuracy
is lower and the training loss is higher than in the no-dropout case, because fewer neurons are active during training. At the
same time, the test loss decreases and the test accuracy improves, especially for dropout = 0.4. This shows that dropout
successfully acts as a regularizer that reduces overfitting and improves generalization, exactly as expected from theory.$\\$

**Answer 2:**$\\$
The low-dropout setting (0.4) provides the best overall performance. In this case, the training loss decreases
steadily, and the test loss reaches the lowest values among all models. Most importantly, the test accuracy is the highest,
showing that the model reaches a good balance between learning useful patterns and avoiding overfitting. This means the model
still has enough capacity to learn, while dropout prevents it from relying too heavily on specific neurons.$\\$

In contrast, the high-dropout setting (0.8) leads to underfitting. The training loss remains high and the training accuracy stays 
very low, because too many neurons are disabled during training. Although the test loss is relatively low due to strong 
regularization, the test accuracy remains limited and is clearly lower than with dropout = 0.4. This shows that while high dropout
prevents overfitting, too much dropout hurts the model's ability to learn meaningful features.$\\$

"""

part2_q2 = r"""
**Your answer:**



"""

part2_q3 = r"""
**Answer 1:**$\\$

Gradient Descent and Stochastic Gradient Descent are both optimization methods used to minimize a loss 
function by updating model parameters in the direction of the negative gradient. 
Bth rely on the same mathematical principle of following the gradient to reduce error, and both aim to find parameter 
values that minimize the risk. In both methods, learning is controlled by a learning rate, and convergence behavior 
is influenced by factors such as model conditioning, loss surface geometry, and step size. $\\$

Gradient Descent and Stochastic Gradient Descent differ mainly in how the gradient is calculated at each step.
In GD, the gradient is calculated using the entire training dataset, meaning that every parameter update is based on the exact 
direction of the total loss. This makes each update very stable and consistent, but also computationally expensive. 
In contrast, SGD estimates the gradient using only a single training example (or a small batch). As a result, each 
update is only an approximation of the true gradient and contains randomness, which makes the updates much cheaper but 
also noisier.$\\$

Another important difference lies in scalability and generalization. GD does not scale well to large datasets because of 
its high computational and memory cost on the other hand the noise in SGD can act as a form of regularization that helps 
the model escape poor local minima and saddle points, often leading to better generalization on unseen data.

**Answer 2:**

Momentum can be applied to Gradient Descent, but its effect is usually limited compared to its 
impact in Stochastic Gradient Descent. Momentum is most useful when the optimization process suffers from 
noisy and unstable updates, since it helps smooth the direction of movement and speeds up progress along 
consistent directions. In full Gradient Descent, however, the gradients are computed over the entire dataset, 
so the updates are already stable and accurate. Because of this, momentum does not have as much to improve, 
and while it can still provide a small speed-up in difficult optimization landscapes, its overall contribution is 
much less significant than in SGD.

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 2  # number of layers (not including output)
    hidden_dims = 128  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    
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

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.01, 1e-4, 0.9  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Answer 1:**$\\$
Optimization error is the error that results from failing to find the best possible parameters for a chosen model, even when
the model is expressive enough to represent the true function. This means we can infer there are imperfections in the training 
process rather than there are limitations in the model itself. This error can be caused by factors such as a poorly chosen 
learning rate, too few training iterations, bad weight initialization, getting stuck in local minima or saddle points, or 
noisy gradient updates.$\\$

Generalization error is the difference between the model's performance on the training data and its performance on unseen data. 
A model may achieve very low training error but still perform poorly on new samples due to overfitting. This error is mainly 
caused by having too complex a model relative to the amount of training data, insufficient regularization, biased or 
non-representative training data, or data leakage from the test set.$\\$

Approximation error is the error that comes from the inherent limitations of the model's structure. If the model class is too 
simple to represent the true underlying relationship in the data, the model will remain inaccurate even with perfect optimization 
and infinite data. This error is caused by using an overly simple hypothesis class, such as a linear model to represent a highly 
nonlinear problem, or by restricting the depth or width of a neural network too severely.$\\$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Answer 1:**
$\\$

Depth = 1:
When the depth is fixed to 1 and the width increases from 2 to 32, the decision boundary becomes slightly more flexible but 
still remains fairly simple. This happens because the network can only create limited nonlinear transformations. with small width, 
the model does not have enough capacity to fit the curve in the data, so the separation is almost linear and the accuracy is lower. 
As the width increases, the model gains more capacity to represent different directions in the data, which allows the boundary to 
bend a bit and improves both validation and test accuracy.

Depth = 2:
With depth equal to 2, increasing the width has a stronger effect because the model now combines features across two layers. 
For small width, the model is still limited and cannot fully represent the curved structure of the classes, but as the width increases, 
the model can form more combinations of features between layers, which makes the decision boundary more nonlinear and better 
aligned with the data. as we can seein the validation and test accuracy which improve more clearly than in the depth 1 case.

Depth = 4 :
At depth 4, as the width increases, the non linearity of the decision boundary grows very quickly, 
and the decision boundary becomes extremely sharp and complex. This happens because many layers with many 
neurons can represent very detailed variations in the data. As a result, the model fits the training and validation data very 
well, which explains the very high validation accuracy. However, this high flexibility also makes the model sensitive to noise, 
so the test accuracy does not always improve at the same rate, indicating an overfitting.

**Answer 2:**
$\\$
Width = 2:
When the width is fixed to 2 and the depth increases, the model remains very narrow. Even though more layers are added, 
each layer has very limited capacity to extract features. This restricts how much information can be passed forward through 
the network. the decision boundary stays close to linear and cannot truly match  to the curved structure of the data. 
As a result, increasing the depth alone does not significantly improve the validation and test accuracy.

Width = 8:
With width equal to 8, the model has more neurons in each layer to learn meaningful features. When the depth increases, 
these features can be transformed multiple times across layers, which allows the network to build more complex representations. 
This is why the decision boundary becomes more curved and accurate, and why both validation and test accuracy improve.

Width = 32
When the width is already very large, increasing the depth gives the model extremely high capacity. Each layer can represent 
many features, and stacking many such layers allows the network to fit very detailed patterns in the data. This explains why 
the validation accuracy becomes very high. However the model can also learn the noise in 
the dataset and be in overfitting, which is why the test accuracy does not increase in the same way and may even slightly decrease. 


"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.01, 1e-4, 0.9  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    
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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""