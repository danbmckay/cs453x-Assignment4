import numpy as np
from cvxopt import solvers, matrix
import sklearn.svm

class SVM453X ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):
        # Add 1s for the bias term
        Xtilde = np.empty([X.shape[0], X.shape[1]+1])
        for i in range(0, len(y)):
            Xtilde[i] = np.hstack((X[i], [1]))
        m, n = Xtilde.shape

        P = np.eye(n)
        q = np.zeros(n)
        G = -y.reshape(-1, 1) * Xtilde
        h = -np.ones(m).reshape(-1, 1)

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        answers = np.array(sol['x'])
        self.w = answers[0:n-1].T
        self.b = np.array([answers[n-1]])




    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        w = self.w
        b = self.b
        ans = np.zeros(x.shape[0])
        for i in range(0, len(x)):
            # x.T * w + b
            temp = x[i].reshape(1,-1).dot(w.T) + b
            if temp > 0: ans[i] = 1
            if temp < 0: ans[i] = -1
        return ans

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm453X = SVM453X()
    svm453X.fit(X, y)
    print("results")
    print(svm453X.w, svm453X.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)

    acc = np.mean(svm453X.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm453X = SVM453X()
    svm453X.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm453X.w) + np.abs(svm.intercept_ - svm453X.b)
    print(diff)

    acc = np.mean(svm453X.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")

if __name__ == "__main__": 
    test1()
    for seed in range(5):
        test2(seed)
