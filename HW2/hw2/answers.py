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
Optimization error is the error that comes from not finding the best possible parameters for a given model, even when the model itself is powerful enough. 
This means the problem is in the training process and not in the model structure. Optimization error can be caused by issues such as a bad learning rate, 
too few training iterations, poor initialization, or the optimizer getting stuck in local minima or saddle points.$\\$

Generalization error is the difference between how well the model performs on the training data and how well it performs on unseen data. A model can achieve 
low training error but still perform poorly on new samples due to overfitting. This usually happens when the model is too complex compared to the amount of data, 
when there is not enough regularization, or when the training data does not represent the test data well.$\\$

Approximation error is the error caused by the limits of the model itself. If the model is too simple to represent the true relationship in the data, 
it will not perform well even if it is trained perfectly. This type of error depends on the model design, such as its depth, width, and choice of architecture.$\\$


**Answer 2:**
Optimization error: 
The optimization error is low. From the loss and accuracy curves, the training loss decreases rapidly and converges to a low value, while the 
training accuracy increases quickly and stabilizes around 93-94%. This indicates that the optimizer is able to find good parameters for the chosen model and is not limited 
by poor convergence or training instability. The decision boundary also supports this conclusion, as it is well-formed and clearly separates the training samples rather 
than appearing random or poorly aligned with the data.

Generalization error:
The model shows a moderate generalization error, which is expected but still noticeable. There is a consistent gap of about 8-12% between training and test accuracy, 
and the test loss remains higher and more unstable than the training loss. From the decision boundary plot, we can see that while the boundary captures the overall 
structure of the data, it also follows some fine details and bends tightly around certain regions. This suggests mild overfitting: the model learns the training data well, 
but some of the learned structure does not transfer perfectly to unseen data.

Approximation error:
The approximation error is low. The decision boundary is clearly nonlinear and matches the general shape of the class distributions, and the model achieves high 
training accuracy with low training loss. This shows that the hypothesis class is expressive enough to represent the underlying pattern in the data. If approximation error 
were high, the boundary would be overly simple and unable to separate the classes well even on the training set, which is not observed here.
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

**Answer 3:**
$\\$
Even though both configurations have approximately the same number of parameters, their behavior is very different. 
The model with depth = 1 and width = 32 creates a smoother nonlinear boundary, but the boundary mostly bends in a single 
direction. This means it can achieve reasonably good performance, but it still cannot model the more complex structure 
present in the data. In contrast, the model with depth = 4 and width = 8 learns a more complex representation. 
Each layer transforms the data further, allowing the decision boundary to capture more detailed nonlinear patterns. 
This leads to higher validation and test accuracy. This comparison shows that, when the number of parameters is similar, 
increasing depth is more effective than simply increasing width for learning complex features.

**Answer 4:**
Yes, choosing the threshold on the validation set helped the test results. From the plots, we can see that the best thresholds chosen on the validation set are not 
always 0.5. This means that using the default threshold would classify too many points to one class. By adjusting the threshold using the validation set, 
the decision boundary is shifted slightly, which leads to fewer classification mistakes. In many configurations, this results in higher test accuracy compared to using 
a fixed threshold. $\\$
However, the improvement is limited. Threshold selection does not change the learned model or the shape of the decision boundary, it only changes where we draw the final 
line between the classes. Therefore, it can correct small biases in the predictions, but it cannot fix underfitting or overfitting caused by the model architecture.


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
**Answer 1:**


Regular block:$\\$
The regular block has two $3\times 3$ convolutional layers that operate directly on a 256-channel input.
which mean that each convolution have:

$$
3 \times 3 \times 256 \times 256 + 256(bias)
= 590,080 \hspace{0.1cm} \text{parameters}
$$

Since there are two layers, the total number of parameters is
$$
2 \times 590{,}080 = 1{,}180{,}160 \hspace{0.1cm} \text{parameters}
$$

Bottleneck block:$\\$
The bottleneck block consists of three convolutions:

First layer: $1\times 1$ conv, $256 \to 64$: $\\$
$$
1 \times 1 \times 256 \times 64 + 64 = 16,448 \hspace{0.1cm} \text{parameters}
$$ $\\$

Second layer: $3\times 3$ conv, $64 \to 64$: $\\$
$$
3 \times 3 \times 64 \times 64 + 64 = 36,928 \hspace{0.1cm} \text{parameters}
$$ $\\$

Third layer: $1\times 1$ conv, $64 \to 256$: $\\$
$$
1 \times 1 \times 64 \times 256 + 256 = 16,640 \hspace{0.1cm} \text{parameters}
$$


Therefore total number of parameters in the bottleneck block is therefore
$$
16,448 + 36,928 + 16,640 = 70,016 \hspace{0.1cm} \text{parameters}
$$

**Answer 2:**

The number of floating point operations can be approximately calculated like this:$\\$

For the regular block:
There are two $3\times 3$ convolutions with $C_{\text{in}} = C_{\text{out}} = 256$, so
$$ 
 2 \times H  \times W \hspace{0.1cm} \text{(spatial input size)} \times 3 \times 3 \hspace{0.1cm} \text{(conv kernal size)} 
 \times 256 \hspace{0.1cm} \text{(input channels)} \times 256 \hspace{0.1cm} \text{(output channels)}
= H \times W \times 1{,}179{,}648.
$$


For the bottleneck block:
$$
\begin{aligned}
\text{First layer:}\quad & H \times W \hspace{0.1cm} \text{(spatial input size)} \times 1 \times 1 \hspace{0.1cm} \text{(conv kernal size)} 
 \times 256 \hspace{0.1cm} \text{(input channels)} \times 64 \hspace{0.1cm} \text{(output channels)}\\
\text{Second layer:}\quad & H \times W \hspace{0.1cm} \text{(spatial input size)} \times 3 \times 3 \hspace{0.1cm} \text{(conv kernal size)} 
 \times 64 \hspace{0.1cm} \text{(input channels)} \times 64 \hspace{0.1cm} \text{(output channels)},\\
\text{Third layer:}\quad & H \times W \hspace{0.1cm} \text{(spatial input size)} \times 1 \times 1 \hspace{0.1cm} \text{(conv kernal size)}
 \times 64 \hspace{0.1cm} \text{(input channels)} \times 256 \hspace{0.1cm} \text{(output channels)}.
\end{aligned}
$$

Suming these, the total Flops are
$$
= H \times W \times 69,632.
$$


**Answer 3:**

Both blocks combine information spatially and across feature maps, but they do so in different ways.

In the regular block, both $3\times 3$ convolutions operate on the full 256-channel representation. 
Stacking these two layers yields an effective receptive field of $5\times 5$, enabling extensive spatial 
aggregation within each feature map. Because all 256 channels are preserved throughout the block, spatial interactions are 
modeled in a high-dimensional space, allowing the block to capture fine spatial structures. This increased 
representational capacity however comes at a high computational and parameter cost.

In the bottleneck block, by contrast, there is only a single $3\times 3$ convolution responsible for spatial 
aggregation, and it operates on a reduced 64-dimensional channel space. Although the effective receptive field 
remains $5\times 5$, spatial mixing occurs in a lower-dimensional representation, making it less expressive spatially 
than the regular block but significantly more efficient.

The bottleneck block compensates for this by using $1\times 1$ convolutions to compress and then re-expand the 
channel dimension (from 256 to 64 and back to 256). These $1\times 1$ layers enable efficient mixing across feature maps, 
allowing the network to reweight, select, and recombine channels while suppressing redundant features. The regular block lacks
this explicit channel compression and expansion mechanism, as both of its layers are purely spatial convolutions operating on 
the full 256-channel space.

to summrize this the regular block is better at spatially combining information within feature maps, while the bottleneck block is 
more effective at combining information across feature maps as a result of $1\times 1$ convolutions.


"""


part4_q2 = r"""
**Your answer:**

Let $\delta_1=\frac{\partial L}{\partial y_1}\in\mathbb{R}^m$. For $y_1=Mx_1$, the Jacobian is $\frac{\partial y_1}{\partial x_1}=M$, hence by the chain rule
$$
\frac{\partial L}{\partial x_1}
=\left(\frac{\partial y_1}{\partial x_1}\right)^\top \frac{\partial L}{\partial y_1}
= M^\top \delta_1.
$$

Now let us look at $k$ such layers. Because in a stacked network, the output of layer $\ell$ becomes the input to layer $\ell+1$,
we will denote the representation after layer $\ell$ by $x^{(\ell)}$ and define
$$
x^{(\ell+1)} = M^{(\ell)} x^{(\ell)}, \qquad \ell=0,1,\dots,k-1.
$$
The loss $L$ depends on the final output $x^{(k)}$, so gradients are passed backward through each layer.

Let $\delta^{(\ell)} := \frac{\partial L}{\partial x^{(\ell)}}$.
For the last layer,
$$
\delta^{(k-1)} = \frac{\partial L}{\partial x^{(k-1)}}
= \left(\frac{\partial x^{(k)}}{\partial x^{(k-1)}}\right)^\top \frac{\partial L}{\partial x^{(k)}}
= \left(M^{(k-1)}\right)^\top \delta^{(k)}.
$$
For one step earlier,
$$
\delta^{(k-2)}
= \left(\frac{\partial x^{(k-1)}}{\partial x^{(k-2)}}\right)^\top \delta^{(k-1)}
= \left(M^{(k-2)}\right)^\top \left(M^{(k-1)}\right)^\top \delta^{(k)}.
$$

Continuing this substitution, each time you move one layer backward you multiply by the transpose of that layer's matrix. After $k$ steps you obtain
$$
\frac{\partial L}{\partial x^{(0)}}
= \delta^{(0)}
= \left(M^{(0)}\right)^\top \left(M^{(1)}\right)^\top \cdots \left(M^{(k-1)}\right)^\top \, \delta^{(k)}
= \Big(\prod_{\ell=0}^{k-1}\left(M^{(\ell)}\right)^\top\Big)\frac{\partial L}{\partial x^{(k)}}.
$$

We now use the inequality $\|AB\|\le \|A\|\,\|B\|$:
$$
\Big\|\frac{\partial L}{\partial x^{(0)}}\Big\|
\le
\Big(\prod_{\ell=0}^{k-1}\big\|\left(M^{(\ell)}\right)^\top\big\|\Big)\Big\|\frac{\partial L}{\partial x^{(k)}}\Big\|
=
\Big(\prod_{\ell=0}^{k-1}\|M^{(\ell)}\|\Big)\Big\|\frac{\partial L}{\partial x^{(k)}}\Big\|.
$$

The condition $|M_{ij}|<1$ indicates small initialization and as the question says that is as in Xavier-style initialization. For the analysis, we can assume this also implies a bound on the operator norm, $\|M^{(\ell)}\|\le \rho<1$. Therefore,
$$
\Big\|\delta^{(0)}\Big\|
\le
\rho^k \Big\|\delta^{(k)}\Big\|
\xrightarrow[k\to\infty]{} 0,
$$
so the gradient magnitude decays (roughly) exponentially with depth.

For the residual form, let $\delta_2=\frac{\partial L}{\partial y_2}$. With
$$
y_2 = x_2 + M x_2 = (I+M)x_2,
$$
the Jacobian is $\frac{\partial y_2}{\partial x_2}=I+M$, hence
$$
\frac{\partial L}{\partial x_2} = (I+M)^\top \delta_2 = \delta_2 + M^\top \delta_2.
$$

The term $\delta_2$ comes from the skip connection and passes backward unchanged, while $M^\top\delta_2$ is the additional part contributed by the weights.
In a stack $x^{(\ell+1)}=(I+M^{(\ell)})x^{(\ell)}$, the backward gradient is multiplied by $(I+M^{(\ell)})^\top$ at each layer
when $\|M^{(\ell)}\|$ is small, these matrices are close to $I$, so the gradient is not forced to shrink like a product of $M^{(\ell)\top}$ terms.





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