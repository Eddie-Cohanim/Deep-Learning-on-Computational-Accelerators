import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        N = x.shape[0]  # Number of samples
        C = x_scores.shape[1]  # Number of classes

        # Get the scores of the correct classes for each sample
        # y is (N,), x_scores is (N, C)
        # We need to gather scores at indices specified by y
        correct_class_scores = x_scores[torch.arange(N), y]  # (N,)

        # Compute the margin matrix M where M[i,j] = s_j - s_{y_i} + delta
        # Broadcast: x_scores is (N, C), correct_class_scores is (N,)
        # We need to reshape correct_class_scores to (N, 1) for broadcasting
        M = x_scores - correct_class_scores.unsqueeze(1) + self.delta  # (N, C)

        # Apply the hinge: max(0, M[i,j])
        M = torch.clamp(M, min=0)

        # Set the loss for the correct class to 0 (we don't want to count it)
        M[torch.arange(N), y] = 0

        # Sum over all classes and samples, then average
        loss = M.sum() / N
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = {'M': M, 'x': x, 'y': y, 'N': N}
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # Retrieve saved context from loss calculation
        M = self.grad_ctx['M']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        N = self.grad_ctx['N']

        # Create matrix G where G[i,j] indicates the gradient coefficient
        # G[i,j] = 1 if M[i,j] > 0 (margin violated for class j)
        # G[i,y_i] = -(number of classes with violated margins for sample i)

        G = (M > 0).float()  # (N, C) - 1 where margin is violated, 0 otherwise

        # For each sample, count how many classes had violated margins
        # This is the sum across classes for each sample
        violated_counts = G.sum(dim=1)  # (N,)

        # Set the gradient for the correct class to be negative of the count
        G[torch.arange(N), y] = -violated_counts

        # Compute gradient: X^T @ G, then average over samples
        grad = (x.T @ G) / N  # (D, C)
        # ========================

        return grad
