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
with each $(m,k)$ slice forming an $(512x1024)$ block.
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
**Answer:**$\\$
Yes, it is possible for test loss to decrease while test accuracy decreases when using 
cross-entropy.
A model can assign higher probability to the correct class without crossing the decision
boundary needed to change the predicted label. In that case the loss goes down because
the probabilities improved, but accuracy stays the same or even gets worse if other 
samples flip from correct to incorrect.

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

**Answer 2:**$\\$

Momentum can be applied to Gradient Descent, but its effect is usually limited compared to its 
impact in Stochastic Gradient Descent. Momentum is most useful when the optimization process suffers from 
noisy and unstable updates, since it helps smooth the direction of movement and speeds up progress along 
consistent directions. In full Gradient Descent, however, the gradients are computed over the entire dataset, 
so the updates are already stable and accurate. Because of this, momentum does not have as much to improve, 
and while it can still provide a small speed-up in difficult optimization landscapes, its overall contribution is 
much less significant than in SGD.$\\$

**Answer 3:**$\\$

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
Optimization error is the error that comes from not finding the best possible parameters for a given model, 
even when the model itself is powerful enough. This means the problem is in the training process and not in 
the model structure. Optimization error can be caused by issues such as a bad learning rate, too few training 
iterations, poor initialization, or the optimizer getting stuck in local minima or saddle points.$\\$

Generalization error is the difference between how well the model performs on the training data and how well 
it performs on unseen data. A model can achieve low training error but still perform poorly on new samples due 
to overfitting. This usually happens when the model is too complex compared to the amount of data, when there 
is not enough regularization, or when the training data does not represent the test data well.$\\$

Approximation error is the error that comes from the inherent limitations of the model's structure. If the model 
class is too simple to represent the true underlying relationship in the data, the model will remain inaccurate 
even with perfect optimization and infinite data. This error is caused by using an overly simple hypothesis 
class, such as a linear model to represent a highly nonlinear problem, or by restricting the depth or width of 
a neural network too severely.$\\$

**Answer 2:**$\\$
Optimization error:$\\$
The optimization error is low. From the loss and accuracy curves, the training loss decreases rapidly
and converges to a low value, while the training accuracy increases quickly and stabilizes around 
93-94%. This indicates that the optimizer is able to find good parameters for the chosen model and 
is not limited by poor convergence or training instability. The decision boundary also supports 
this conclusion, as it is well-formed and clearly separates the training samples rather than 
appearing random or poorly aligned with the data.$\\$

Generalization error:$\\$
The model shows a moderate generalization error, which is expected but still noticeable. There is a 
consistent gap of about 8-12% between training and test accuracy, 
and the test loss remains higher and more unstable than the training loss. From the decision boundary plot, we can see that while the boundary captures the overall 
structure of the data, it also follows some fine details and bends tightly around certain regions. This suggests mild overfitting: the model learns the training data well, 
but some of the learned structure does not transfer perfectly to unseen data.$\\$

Approximation error:$\\$
The approximation error is low. The decision boundary is clearly nonlinear and matches the general 
shape of the class distributions, and the model achieves high training accuracy with low training loss.
This shows that the hypothesis class is expressive enough to represent the underlying pattern in the 
data. If approximation error were high, the boundary would be overly simple and unable to separate 
the classes well even on the training set, which is not observed here.$\\$
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
**Answer 1:**
$\\$

Depth = 1:$\\$
When the depth is fixed to 1 and the width increases from 2 to 32, the decision boundary becomes 
slightly more flexible but still remains fairly simple. This happens because the network can only 
create limited nonlinear transformations. with small width, the model does not have enough capacity 
to fit the curve in the data, so the separation is almost linear and the accuracy is lower. 
As the width increases, the model gains more capacity to represent different directions in 
the data, which allows the boundary to bend a bit and improves both validation and test accuracy.$\\$

Depth = 2:$\\$
With depth equal to 2, increasing the width has a stronger effect because the model now combines 
features across two layers. For small width, the model is still limited and cannot fully represent 
the curved structure of the classes, but as the width increases, the model can form more combinations 
of features between layers, which makes the decision boundary more nonlinear and better aligned with 
the data. as we can seein the validation and test accuracy which improve more clearly than in the depth
1 case.$\\$

Depth = 4 :$\\$
At depth 4, as the width increases, the non linearity of the decision boundary grows very quickly, 
and the decision boundary becomes extremely sharp and complex. This happens because many layers with 
many neurons can represent very detailed variations in the data. As a result, the model fits the 
training and validation data very well, which explains the very high validation accuracy. However, 
this high flexibility also makes the model sensitive to noise, so the test accuracy does not always 
improve at the same rate, indicating an overfitting.$\\$

**Answer 2:**
$\\$
Width = 2:$\\$
When the width is fixed to 2 and the depth increases, the model remains very narrow. Even though more 
layers are added, each layer has very limited capacity to extract features. This restricts how much 
information can be passed forward through the network. the decision boundary stays close to linear 
and cannot truly match  to the curved structure of the data. As a result, increasing the depth alone 
does not significantly improve the validation and test accuracy.$\\$

Width = 8:$\\$
With width equal to 8, the model has more neurons in each layer to learn meaningful features. When 
the depth increases, these features can be transformed multiple times across layers, which allows 
the network to build more complex representations. This is why the decision boundary becomes more 
curved and accurate, and why both validation and test accuracy improve.$\\$

Width = 32:$\\$
When the width is already very large, increasing the depth gives the model extremely high capacity. 
Each layer can represent many features, and stacking many such layers allows the network to fit 
very detailed patterns in the data. This explains why the validation accuracy becomes very high. 
However the model can also learn the noise in the dataset and be in overfitting, which is why the 
test accuracy does not increase in the same way and may even slightly decrease.$\\$

**Answer 3:**
$\\$
Even though both configurations have approximately the same number of parameters, their behavior is 
very different. The model with depth = 1 and width = 32 creates a smoother nonlinear boundary, 
but the boundary mostly bends in a single direction. This means it can achieve reasonably good 
performance, but it still cannot model the more complex structure present in the data. In contrast,
the model with depth = 4 and width = 8 learns a more complex representation. Each layer transforms 
the data further, allowing the decision boundary to capture more detailed nonlinear patterns. 
This leads to higher validation and test accuracy. This comparison shows that, when the number of 
parameters is similar, increasing depth is more effective than simply increasing width for learning 
complex features.$\\$

**Answer 4:**
$\\$
Yes, choosing the threshold on the validation set helped the test results. From the plots, we can see 
that the best thresholds chosen on the validation set are not always 0.5. This means that using the 
default threshold would classify too many points to one class. By adjusting the threshold using the 
validation set, the decision boundary is shifted slightly, which leads to fewer classification 
mistakes. In many configurations, this results in higher test accuracy compared to using a fixed 
threshold. $\\$
However, the improvement is limited. Threshold selection does not change the learned model or the 
shape of the decision boundary, it only changes where we draw the final line between the classes. 
Therefore, it can correct small biases in the predictions, but it cannot fix underfitting or 
overfitting caused by the model architecture.$\\$

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
**Answer 1:**$\\$

Regular block:$\\$
The regular block has two $3\times 3$ convolutional layers that operate directly on a 256-channel 
input. which mean that each convolution have:

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


**Answer 3:**$\\$

Both blocks combine information spatially and across feature maps, but they do so in different
ways.$\\$

In the regular block, both $3\times 3$ convolutions operate on the full 256-channel representation. 
Stacking these two layers yields an effective receptive field of $5\times 5$, enabling extensive 
spatial aggregation within each feature map. Because all 256 channels are preserved throughout the 
block, spatial interactions are modeled in a high-dimensional space, allowing the block to capture 
fine spatial structures. This increased representational capacity however comes at a high 
computational and parameter cost.$\\$

In the bottleneck block, by contrast, there is only a single $3\times 3$ convolution responsible 
for spatial aggregation, and it operates on a reduced 64-dimensional channel space. Although the 
effective receptive field remains $5\times 5$, spatial mixing occurs in a lower-dimensional 
representation, making it less expressive spatially than the regular block but significantly 
more efficient.$\\$

The bottleneck block compensates for this by using $1\times 1$ convolutions to compress and 
then re-expand the channel dimension (from 256 to 64 and back to 256). These $1\times 1$ layers 
enable efficient mixing across feature maps, allowing the network to reweight, select, and recombine
channels while suppressing redundant features. The regular block lacks this explicit channel 
compression and expansion mechanism, as both of its layers are purely spatial convolutions operating 
on the full 256-channel space.$\\$

to summrize this the regular block is better at spatially combining information within feature maps, 
while the bottleneck block is more effective at combining information across feature maps as a 
result of $1\times 1$ convolutions.$\\$

"""


part4_q2 = r"""
**Your answer:**$\\$

Let $\delta_1=\frac{\partial L}{\partial y_1}\in\mathbb{R}^m$. For $y_1=Mx_1$, the Jacobian is
$\frac{\partial y_1}{\partial x_1}=M$, hence by the chain rule
$$
\frac{\partial L}{\partial x_1}
=\left(\frac{\partial y_1}{\partial x_1}\right)^\top \frac{\partial L}{\partial y_1}
= M^\top \delta_1.
$$

Now let us look at $k$ such layers. Because in a stacked network, the output of layer $\ell$ becomes 
the input to layer $\ell+1$, we will denote the representation after layer $\ell$ by $x^{(\ell)}$ 
and define
$$
x^{(\ell+1)} = M^{(\ell)} x^{(\ell)}, \qquad \ell=0,1,\dots,k-1.
$$
The loss $L$ depends on the final output $x^{(k)}$, so gradients are passed backward through each 
layer.$\\$

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

Continuing this substitution, each time you move one layer backward you multiply by the transpose 
of that layer's matrix. After $k$ steps you obtain
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

The condition $|M_{ij}|<1$ indicates small initialization and as the question says that is as in 
Xavier-style initialization. For the analysis, we will assume this also implies a bound on the 
operator norm, $\|M^{(\ell)}\|\le \rho<1$. Therefore,
$$
\Big\|\delta^{(0)}\Big\|
\le
\rho^k \Big\|\delta^{(k)}\Big\|
\xrightarrow[k\to\infty]{} 0,
$$
so the gradient magnitude decays exponentially with depth.

For the residual form, let $\delta_2=\frac{\partial L}{\partial y_2}$. With
$$
y_2 = x_2 + M x_2 = (I+M)x_2,
$$
the Jacobian is $\frac{\partial y_2}{\partial x_2}=I+M$, hence
$$
\frac{\partial L}{\partial x_2} = (I+M)^\top \delta_2 = \delta_2 + M^\top \delta_2.
$$

The term $\delta_2$ comes from the skip connection and passes backward unchanged, while 
$M^\top\delta_2$ is the additional part contributed by the weights. In a stack 
$x^{(\ell+1)}=(I+M^{(\ell)})x^{(\ell)}$, the backward gradient is multiplied by 
$(I+M^{(\ell)})^\top$ at each layer when $\|M^{(\ell)}\|$ is small, these matrices are close 
to $I$, so the gradient is not forced to shrink like a product of $M^{(\ell)\top}$ terms.$\\$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Answer 1:**$\\$

The results show a clear non-monotonic relationship between network depth and accuracy.
For both K=32 and K=64, the shallower networks (L=2 and L=4) train successfully and achieve 
reasonable accuracy, while very deep networks (L=8,16) completely fail to train.$\\$

However, for K=32 and K=64 the test accuracy completly collapses to ~10% 
(random guessing for 10-class CIFAR-10) at L=8 and L=16 and for K=32 peaks at L=2 with 61.7%, 
then drops slightly to 57.7% at L=4, whereas for K=64 the optimal depth is L=4 with 67.6% test
accuracy.$\\$

This means that the best results are achieved at moderate depths: L=2 for K=32 and L=4 for K=64.
This makes sense because these networks are deep enough to extract hierarchical features but 
shallow enough to avoid optimization problems. The deeper network (L=4) works better 
with more filters (K=64) because the additional representational capacity helps compensate 
for the increased depth.$\\$

**Answer 2:**$\\$

Yes, the networks with L=8 and L=16 were completely untrainable. Both networks achieved only
~10% test accuracy, which is equal to random guessing on the 10-class CIFAR-10 dataset.
The loss remained stuck at approximately 2.303 (which is -ln(1/10), the loss for uniform 
random predictions) throughout training, and training was terminated early due to no improvement.$\\$

This failure is caused by the vanishing gradient. In very deep networks without skip 
connections, gradients must be backpropagated through many layers. Since each layer applies 
an activation function (ReLU) and potentially pooling, the gradients get multiplied by many 
values less than 1 during backpropagation. By the time gradients reach the early layers, 
they become extremely small (vanish), making it impossible for those layers to learn 
meaningful features. Without proper features in early layers, the entire network cannot learn.
We can solve this in 2 ways.$\\$

1. Use residual or skip connections in a ResNet-style architecture. These connections
let gradients pass through the network more directly, instead of being repeatedly 
transformed by activation functions. This creates clear gradient pathways that make 
training much deeper models feasible and help mitigate the vanishing gradient problem.$\\$

2. Add batch normalization layers after each convolution to keep activations well behaved and 
training stable. By normalizing activations, the network avoids extreme values, which helps 
gradients stay informative as they propagate through the layers and makes deeper models easier 
to train.$\\$

"""

part5_q2 = r"""
**Answer:**$\\$

The results show that changing the number of filters does not have a simple or uniform effect
on performance. Compared to the earlier experiment that focused on depth, this makes it clear 
that model capacity is not just about adding more layers or more filters. The interaction 
between depth and width matters.$\\$

For shallow networks with L=2, increasing the number of filters hurts performance. 
This suggests that when the network is too shallow, adding parameters does not help it learn 
better and may even make optimization or generalization harder.$\\$

When L=4, the expected behavior appears. As the number of filters increases, performance 
improves. At this depth, the network can take advantage of the extra capacity and learn more
complicated features.$\\$

When the network becomes deeper, that means when L=8, all configurations fail regardless 
of how many filters are used. Performance becoms similar to random guessing, showing that 
training breaks down completely. In this case, the vanishing gradient problem causes the model
to not learn at all, and increasing width cannot compensate.$\\$

Taken together, the experiments show that making a network very deep does not work, even if it 
is very wide. Depth and width work best when they are balanced. Networks with moderate depth 
can benefit from more filters, while shallow networks cannot. Overall, good performance 
comes from a balanced design rather than pushing depth or width to extremes.$\\$

"""

part5_q3 = r"""
**Answer:**$\\$

In this experiment,the later layers have more filters than the earlier ones. 
even though the networks themselves are not very deep.
The idea is to let earlier layers focus on simpler features and later 
layers handle more complex ones, which is a common CNN design choice.$\\$

With two and three layers, this approach works well and gives better results 
than using the same number of filters in every layer. Increasing the filters 
in the later layers helps the model use its capacity more effectively, even at 
small depths.$\\$

However, the limitation is still clear. When the network reaches four layers, 
training fails despite the new filter sizes. This shows that increasing filters 
across layers helps at small depths, but it cannot overcome training issues caused 
by adding more layers. Depth remains the main bottleneck without skip connections.$\\$

"""

part5_q4 = r"""
**Answer:**$\\$

This experiment shows that skip connections make a major difference when training deeper networks.
With ResNet, networks with eight or sixteen layers are able to train successfully, whereas in 
experiment 1.1 models at these depths failed due to training breakdown and vanishing gradiants.$\\$

Compared to experiment 1.3, the improvement is still clear, though not as big.
In experiment 1.4, networks with two, four, and eight layers all train better than before, 
benefiting from both skip connections and the use of more filters. While experiment 1.3 already 
showed that additional filters can work well at low depth, adding skip connections allows 
the model to train more reliably and extract better features, thus improving results.$\\$

Overall, the biggest gain is seen when depth increases. Skip connections turn depth from a 
limitation/hindernce into an advantage/improvment, making deeper networks trainable and 
improving performance across all tested depths.$\\$

"""
part5_q5 = r"""
**Answer:**$\\$

YourCNN adds skip connections to help train deeper networks. Every P=8 layers are grouped
with a skip connection that lets data jump ahead. This helps gradients flow backward during training.
We also use batch normalization, dropout (p=0.2), K=[32, 64, 128] filters, and early stopping=6.$\\$

Results comparing old CNN vs YourCNN with skip connections:$\\$

- L=3: 68.4% vs 71.8% (skip connections slightly worse)
- L=6: 69.7% vs 66.0% (skip connections better)
- L=9: 72.7% vs 63.9% (skip connections much better)
- L=12: 69.3% vs 31.5% (old CNN failed completely due to vanishing gradients)

At L=3, skip connections don't help because the network is shallow enough to train anyway.
At L=6 and L=9, skip connections start improving results. At L=12, the difference is huge -
the old CNN suffered from vanishing gradients and barely learned anything while YourCNN
with skip connections still trains well.$\\$

Best accuracy: 72.7% at L=9. Skip connections make it possible to train deep networks.$\\$

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