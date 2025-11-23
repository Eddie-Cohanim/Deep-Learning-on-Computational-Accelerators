r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**1. False.** Not every split into two disjoint subsets is equally useful.  
For example, if we use a \(10\%\) training set and a \(90\%\) test set, the model will  
perform poorly because it does not have enough data to learn the underlying patterns.  

In addition, even if the split sizes are reasonable, a split can still be useless if it
does not represent the full data distribution. For example, in a binary classification
task with classes "CAT" and "DOG", it is possible that the test set accidentally contains  
only class "CAT". in that case, the model is never trained on class "DOG", therefore leading
to poor results

**2. False.** Cross-validation is performed only on the training set to prevent any leakage from
the test set. In this procedure, the training data is split into K folds. Each fold takes a turn
serving as the validation set, while the remaining K-1 folds are used for training. This setup lets
the model be evaluated on data it hasn't seen during each training run, but still keeps the test set
untouched. The test set is used only at the very end to assess the final model's performance on 
unseen data. (Cross-validation provides a more reliable way to tune hyperparameters than relying on
a single validation split.)

**3. True.** During cross-validation, the model is trained using only a portion of the training data
while the remaining fold is kept completely separate and used as a validation set. Because the model has
never seen the data in that validation fold during training, its performance on that fold serves as an
unbiased estimate of how well it can generalize to unseen data. This setup mimics the role of a test set,
but in a controlled, repeated manner.
Each fold rotates into the role of the validation set once, ensuring that every data point is used both for
training and for validation across different runs. After evaluating the model on all K validation folds,
the performance metrics are averaged. This reduces the variance that could arise from relying on a single
train-validation split and provides a more stable, trustworthy estimate of the model's true predictive ability.
Overall, cross-validation strengthens model evaluation by using more of the data effectively, limiting overfitting
risks, and giving a clearer picture of how the model will behave on completely unseen examples.

**4. True** Adding noise to the training set can help validate whether or no tthe model is robust.
if the model still accuratly predicts the correct results even though we added noise to the
training set, then we know that it inst drastically affected by small mistakes/changes to the data set 

"""

part1_q2 = r"""
**NO.** My friend's approach is not justifiable because he used the test set when choosing the value of $\lambda$
By doing so, he allowed information from the test set to influence the model selection process. The correct
procedure is to tune hyperparameters using only the training data typically through cross validation where the model
is trained on a subset of the data and evaluated on a validation fold that it has not seen during training.
This prevents any leakage of test set information.

Once hyperparameters such as $\lambda$ are chosen using cross validation, the test set should be used exactly once 
at the very end to provide an unbiased estimate of the model's generalization ability. If we evaluate multiple models
on the test set while choosing the best parameters, we risk overfitting to that test set, making the selected model look
better than it truly is on new data. Therefore, the test set must remain completely independent during model development
and be used only for the final evaluation.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
In the structured hinge loss, $\Delta$ represents the required margin between the score of the correct class and the scores
of incorrect classes. When $\Delta > 0$, the model is penalized whenever an incorrect class score comes too close to the 
correct class score. This enforces a positive margin and encourages the model to maintain a clear separation between classes.
The regularization term still pushes the weights to be smaller, which promotes generalization but can make it harder for the model
to maintain a very large margin.
If we allow $\Delta < 0$, the meaning of the margin constraint changes. A negative $\Delta$ effectively allows the score of the
correct class to be lower than the scores of incorrect classes by up to$|\Delta|$. Since the regularization term penalizes large
weights, the model can minimize the overall loss by shrinking the weights toward zero. When all weights collapse to zero, all
class scores become identical, and the model reaches a trivial solution $W = 0$
This leads to poor generalization and many misclassifications, since the margin requirement no longer enforces the correct
class to be preferred.

"""

part2_q2 = r"""
By examining the weight visualizations, we see that the model learns which pixel regions are important for identifying each digit.
Bright regions correspond to positive weights, meaning the model increases the score for a digit when pixels in those areas are activated.
Dark regions correspond to negative weights, which penalize the presence of pixels in those regions. This effectively forms a map
of pixel importance: the model assigns high weight to strokes and shapes that commonly appear in a digit, and negative weight to regions 
where pixels should not appear. For example, in the digit 0, the model expects a dark central region, reflecting the hollow center
characteristic of zeros. White pixels appearing in that region would significantly lower the score. Because some digits share 
geometric similarities, such as 7 and 9, 4 and 9 or 5 and 6. therefore variations in handwriting may activate weight patterns associated
with the wrong digit, resulting in misclassification.

From the visualization of the learned features, we can also infer that the model captures the main structural characteristics
of each digit rather than memorizing exact pixel arrangements. For instance, the digit 5 often has a disconnected bottem left and a strong
vertical stroke on the left, while the digit 6 usually has a connected circular bottom region with a downward curve above. These consistent 
visual cues help the model distinguish between digits by focusing on edges, curves, and connectivity patterns that define each one.

"""

part2_q3 = r"""
**Your answer:**
Based on the training loss graph, the chosen learning rate appears to be good. The loss decreases smoothly and consistently 
throughout the epochs without oscillations or divergence, which indicates stable learning. If the learning rate were too low,
the loss curve would decrease very slowly and appear almost flat. If it were too high, the curve would show instability with
sharp jumps or even increases in loss due to overshooting.

From the accuracy graph, the model is slightly overfitted to the training set. The training accuracy is consistently a bit higher
than the test accuracy, although both curves remain close and follow a similar trend. This small gap indicates mild overfitting.
A highly overfitted model would show a large accuracy gap, while an underfitted model would have low accuracy on both the training
and test sets.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal residual plot would show all errors equal to zero, which would appear as a perfectly straight horizontal line at the zero
level. In that case, the MSE would also be 0. Of course, this rarely happens in practice, but it serves as the
theoretical benchmark. The trained model appears to fit the data well, as both the training and test points cluster closely and
follow a similar pattern. The residual plot after cross validation indicates that the final model performs better than the
model that was restricted to the top five features. The residuals are more tightly concentrated around the zero line and show a more
consistent pattern. This improvement is supported by a lower MSE. In addition, the final model produces noticeably fewer outliers in
the residuals, suggesting better overall predictive stability.

"""

part3_q2 = r"""
**Your answer:**
**1.**
It's still a linear regression model because its still linear in its parameters.
Even if we apply a non-linear feature mapping such as
$$
x \;\mapsto\; (x,\; x^2,\; x^3),
$$
the model remains a linear combination of these transformed features. 

**2.** 
We can apply any non-linear transformation to the original features, but this does not
mean we can fit any non-linear function. The model can only represent functions that
lie in the span of the specific feature mappings we choose. Non-linear features are
useful only when they capture meaningful structure in the data and help separate or
explain the patterns we want to model.

At the same time, adding many non-linear features increases the dimensionality of the*
feature space, which can lead to overfitting

**3.**
In the transformed feature space, the decision boundary is still a hyperplane, since
the classifier remains a linear function of the new features. The non-linearity comes
from the feature mapping, not from the model itself. When we look back at the boundary
in the original feature space, it no longer appears linear. Instead, the mapping can
bend or curve the hyperplane, producing a non-linear decision boundary in the
original input space. In this way, linear models gain the ability to separate data
that is not linearly separable in the raw feature space.

"""

part3_q3 = r"""
**Your answer:**


**1.** We want to compute
$$ \mathbb{E}[|y-x|] = \int_{0}^{1}\int_{0}^{1} |y-x|\, dx\, dy $$
where $x,y \sim \mathrm{Uniform}(0,1)$

let us split the domain according to the absolute value options:
$
|y-x| =
\begin{cases}
y-x, & y \ge x,\\[4pt]
x-y, & x \ge y
\end{cases}
$

therefore we are getting,
$$
\mathbb{E}[|y-x|]
= \int_{0}^{1}\int_{0}^{y} (y-x)\, dx\, dy
+ \int_{0}^{1}\int_{0}^{x} (x-y)\, dy\, dx
$$

we start by computing the first integral:
$$
\int_{0}^{1}\int_{0}^{y} (y-x)\, dx\, dy
= \int_{0}^{1} \left( yx - \frac{x^{2}}{2} \right)_{x=0}^{x=y} dy
= \int_{0}^{1} \frac{y^{2}}{2}\, dy
= \frac{1}{6}
$$

then we continiue with the second integral:
$$
\int_{0}^{1}\int_{0}^{x} (x-y)\, dy\, dx
= \int_{0}^{1} \left( xy - \frac{y^{2}}{2} \right)_{y=0}^{y=x}= \int_{0}^{1} \frac{x^{2}}{2}\, dx
= \frac{1}{6}$$

Therefore,
$$
\mathbb{E}[|y-x|] = \text{first integral} + \text{second integral} =  \frac{1}{6} + \frac{1}{6} = \frac{1}{3}
$$

**2.**

Since $x \sim \mathrm{Uniform}(0,1)$, we compute
$$
\mathbb{E}_x[|\hat{x}-x|]
= \int_0^1 |\hat{x}-x|\,dx
= \int_0^{\hat{x}} (\hat{x}-x)\,dx + \int_{\hat{x}}^1 (x-\hat{x})\,dx
$$

The first integral is
$$
\int_0^{\hat{x}} (\hat{x}-x)\,dx
= \left[\hat{x}x - \frac{x^2}{2}\right]_0^{\hat{x}}
= \frac{\hat{x}^2}{2}
$$

The second integral is
$$
\int_{\hat{x}}^1 (x-\hat{x})\,dx
= \left[\frac{x^2}{2} - \hat{x}x\right]_{\hat{x}}^1
= \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}
$$

Adding both terms:
$$
\mathbb{E}_x[|\hat{x}-x|]
= \hat{x}^2 - \hat{x} + \frac{1}{2}
$$

**3.** 

A constant term in a polynomial can be dropped when optimizing because it does not
affect the minimizer. If
$$
L(\hat{x}) = \hat{x}^2 - \hat{x} + \frac12
$$
and the derivative is
$$
L'(\hat{x}) = 2\hat{x} - 1
$$

and as we can easly see the constant part didnt contribute at all.

"""

# ==============

# ==============
