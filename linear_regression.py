import numpy as np

class Linreg_MSE:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):     
        n, m = X.shape
        
        X_train = X
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        # w = (XT*X)−1*XT*Y
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y 

        return self
        
    def predict(self, X):
        n, m = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w
        return y_pred
    
    def get_weights(self):
        return self.w

class Linreg_Gradient(Linreg_MSE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = None
    
    def fit(self, X, y, lr=0.01, max_iter=100):
        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)
        
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        
        self.losses = []
        
        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(MSE(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= lr * grad

        return self

    def _calc_gradient(self, X, y, y_pred):
        grad = 2 * (y_pred - y)[:, np.newaxis] * X
        grad = grad.mean(axis=0)
        return grad

    def get_losses(self):
        return self.losses

class Linreg_SGD(Linreg_Gradient):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.n_sample = n_sample

    def _calc_gradient(self, X, y, y_pred):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)

        grad = 2 * (y_pred[inds] - y[inds])[:, np.newaxis] * X[inds]
        grad = grad.mean(axis=0)

        return grad

#AdaGrad
class Linreg_AdaGrad(Linreg_SGD):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.n_sample = n_sample

    def fit(self, X, y, lr=0.01, max_iter=100, eps = 1e-6):
        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)
        
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        
        gti = np.zeros(n)

        self.losses = []
        
        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(MSE(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"


            gti += grad**2
            adj_grad = grad / np.sqrt(gti + eps)
            self.w -= lr * adj_grad

        return self

#RMSProp
class Linreg_RMSProp(Linreg_SGD):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.n_sample = n_sample

    def fit(self, X, y, lr=0.01, max_iter=100, beta = 0.6, eps = 1e-8):
        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)
        
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        
        gti = np.zeros(n)

        self.losses = []
        
        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(MSE(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"

            gti = beta * gti + (1-beta) * grad**2
            adj_grad = grad / np.sqrt(gti + eps)
            self.w -= lr * adj_grad

        return self

#Adam
class Linreg_Adam(Linreg_SGD):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.n_sample = n_sample

    def fit(self, X, y, lr=0.01, max_iter=100, beta1 = 0.9, beta2 = 0.99, eps = 1e-8):
        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)
        
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        
        m_w = np.zeros(n)
        v_w = np.zeros(n)

        self.losses = []
        
        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(MSE(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"

            m_w = beta * m_w + (1-beta) * grad
            v_w = beta * v_w + (1-beta) * grad**2
            m_w /= 1-beta1**(iter_num+1)
            v_w /= 1-beta2**(iter_num+1)
            self.w -= lr * m_w/np.sqrt(v_w + eps)

        return self

#L1
class Linreg_L1(Linreg_SGD):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.alpha = alpha

    def _calc_gradient(self, X, y, y_pred):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)

        signw = np_soft_sign(self.w)
        if self.fit_intercept:
            signw[-1] = 0

        grad = X[inds].T @ (y_pred[inds] - y[inds])[:, np.newaxis] / self.n_sample
        grad += self.alpha * signw[:, np.newaxis]

        return grad.flatten()

#L2
class Linreg_L2(Linreg_SGD):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs) 
        self.w = None
        self.alpha = alpha

    def _calc_gradient(self, X, y, y_pred):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)

        lambdaI = self.alpha * np.eye(self.w.shape[0])
        if self.fit_intercept:
            lambdaI[-1, -1] = 0

        grad = 2 * (X[inds].T @ X[inds] / self.n_sample + lambdaI) @ self.w
        grad -= 2 * X[inds].T @ y[inds] / self.n_sample

        return grad

# MSE
def MSE(pred, test):
    return np.mean((pred-test)**2)


#R2
def R2(pred, test):
    mean_test = np.full_like(test, np.mean(test))
    return 1 - np.mean((pred - test)**2)/np.mean((pred - mean_test)**2)

