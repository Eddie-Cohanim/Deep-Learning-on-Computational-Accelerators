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
**NO.** My friend's approach is not justifiable because he used the test set to choose the value of $\lambda$
By doing that, he allowed the test set to influence the model selection process, instead of using a separate
validation set. As a result, the test set is no longer an unbiased and independent estimate of the model's
 generalization performance.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
If we allow $\Delta < 0$ in the SVM loss, the margin constraint becomes reversed. Instead of 
requiring the correct class score to be higher than the incorrect class scores, the loss now 
allows the correct class score to be lower by up to $|\Delta|$.

Since the regularization term punishes large weights, the model can minimize the full loss 
by shrinking all weights toward zero. In this case all class scores become equal,
and the model collapses to the solution $W = 0$. 

"""

part2_q2 = r"""
From the visualization, we can assume that the model is learning the main visual characteristics
of each digit. For example, in the digit 4, the top part is usually not connected and there is a long
vertical line on the right side of the sketch. In contrast, the digit 9 typically has a connected circular
top part and a downward stroke.

This suggests that the model is not simply memorizing pixel patterns, but instead learning the prominent
visual features or strokes that consistently define each number, such as edges, curves, junctions,
and whether certain parts are connected or disconnected.
In other words, the model learns the most distinctive characteristics of each digit and then uses these
learned patterns to correctly identify the digits in the test set.

"""

part2_q3 = r"""
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
# Part 3 answers

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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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

# ==============
