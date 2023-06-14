"""This code is copy-pasted from the DAGMA repository
https://github.com/kevinsbello/dagma
All credit belongs to the original authors.
"""


import igraph as ig
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm

from data_centric_synth.data_models.experiments_common import (
    CausalPerformanceEvaluation,
)


class DAGMA_linear:
    def __init__(self, loss_type, verbose=False, dtype=np.float64):
        super().__init__()
        losses = ["l2", "logistic"]
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def _score(self, W):
        """Evaluate value and gradient of the score function."""
        if self.loss_type == "l2":
            dif = self.Id - W
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
        elif self.loss_type == "logistic":
            R = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
        return loss, G_loss

    def _h(self, W, s=1.0):
        """Evaluate value and gradient of the logdet acyclicity constraint."""
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W, mu, s=1.0):
        """Evaluate value of the penalized objective function."""
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad**2)
        m_hat = self.opt_m / (1 - beta_1**iter)
        v_hat = self.opt_v / (1 - beta_2**iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

    def minimize(
        self,
        W,
        mu,
        max_iter,
        s,
        lr,
        tol=1e-6,
        beta_1=0.99,
        beta_2=0.999,
        pbar=None,
    ):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(
            f"\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations",
        )

        for iter in range(1, max_iter + 1):
            ## Compute the (sub)gradient of the objective
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):  # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f"W went out of domain for s={s} at iteration {iter}")
                    return W, False
                else:
                    W += lr * grad
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f"Learning rate decreased to lr: {lr}")

            if self.loss_type == "l2":
                G_score = -mu * self.cov @ (self.Id - W)
            elif self.loss_type == "logistic":
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f"\nInner iteration {iter}")
                self.vprint(f"\th(W_est): {h:.4e}")
                self.vprint(f"\tscore(W_est): {score:.4e}")
                self.vprint(f"\tobj(W_est): {obj_new:.4e}")
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True

    def fit(
        self,
        X,
        lambda1,
        w_threshold=0.3,
        T=5,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7, 0.6],
        warm_iter=3e4,
        max_iter=6e4,
        lr=0.0003,
        checkpoint=1000,
        beta_1=0.99,
        beta_2=0.999,
    ):
        ## INITALIZING VARIABLES
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        if self.loss_type == "l2":
            self.X = self.X - X.mean(axis=0, keepdims=True)

        self.cov = X.T @ X / float(self.n)
        self.W_est = np.zeros((self.d, self.d)).astype(
            self.dtype,
        )  # init W0 at zero matrix
        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                self.vprint(
                    f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}",
                )
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        ## START DAGMA
        with tqdm(total=(T - 1) * warm_iter + max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f"\nIteration -- {i+1}:")
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(
                        self.W_est.copy(),
                        mu,
                        inner_iters,
                        s[i],
                        lr=lr_adam,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        pbar=pbar,
                    )
                    if success is False:
                        self.vprint("Retrying with larger s")
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                mu *= mu_factor

        ## Store final h and score values and threshold
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


# originally called "count_accuracy" in the dagma repo
def evaluate_dag_model(B_true, B_est) -> CausalPerformanceEvaluation:
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (-1 == B_est.T)).any():
            raise ValueError("undirected edge should only appear once")
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")
        if not is_dag(B_est):
            print("B_est is not a DAG somehow")
            # TODO why does this happen?
            # raise ValueError("B_est should be a DAG")
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return CausalPerformanceEvaluation(
        fdr=fdr,
        tpr=tpr,
        fpr=fpr,
        shd=shd,
        nnz=pred_size,
    )
