# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SolveMinProb:
    """
    Class SolveMinProb includes three subclasses/algorityms: LLS (Linear Least Squares),
    mini batch gradient algorithm and stochastic gradient algorithm with adam. These algorithms are used to find
    the optimum solution w_hat to the minimization problem min_w ||y-Aw||^2.
    
    Inputs
    ------
    y : column vector with the regressand (Np rows)
        float
    A : Ndarray (Np rows Nf columns) with the Nf regressors for each of the Np points
        float

    Attributes
    ----------
    matr : Ndarray A, with the measured data (Np rows, Nf columns)
        float
    vect : y, Ndarray of known values of the regressand, shape (Np,)
        float
    sol : w, Ndarray with shape (Nf,), optimum value of vector w that minimizes ||y-Aw||^2
        float
    Np : number of available points (number of rows of A and y)
        integer
    Nf : number of available features (number of columns of A)
        integer
    min : value of ||y-Aw_hat||^2
        float
    err : Ndarray that stores the iteration step and corresponding value of ||y-Aw||^2 with the current w
        float   
        
    Methods
    -------
    plot_err : plots the value of the objective function as a function of the iteration step
                Parameters: title (default 'Square error'), 
                            logy (0 for linear scale 1 for log scale on the y axis)
                            logx (0 for linear scale 1 for log scale on the x axis)
    print_results : prints the optimum weights w_hat and the corresponding objective function
    plot_w_hat: plots the solution w_hat
    
    """

    def __init__(self, y, A):
        self.matr = A  # matrix with the measured data: Np rows/points and Nf columns/features
        self.Np = y.shape[0]  # number of points
        self.Nf = A.shape[1]  # number of features/regressors
        self.vect = y  # regressand
        self.sol = np.zeros((self.Nf,), dtype=float)  # unknown optimum weights w_hat (Nf elements)
        self.min = 0.0  # obtained minimum value ||y-Aw_hat||^2
        self.err = np.zeros((100, 2), dtype=float)  # in case of iterative minimization, self.err
        # stores the value of the function to be minimized, the first column stores the
        # iteration step, the second the corresponding value of the function
        return

    def plot_err(self, title='', logy=0, logx=0):
        """ 
        plot_err plots the function to be minimized, at the various iteration steps

        Parameters
        ----------
        title : title of the plot
            string
        logy : 1 for logarithmic y scale, 0 for linear y scale
            integer
        logx : 1 for logarithmic x scale, 0 for linear x scale
            integer
        """
        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title+' Square error')
        plt.margins(0.01, 0.1)  # leave some space between the max/min value and the frame of the plot
        plt.grid()
        plt.savefig(f'./{title}-err.png')
        plt.show()
        return

    def print_result(self, title):
        """ 
        print_result prints a first line (title) and then the minimization results:
        the optimum weight vector and the found minimum

        Parameters
        ----------
        title : typically the algorithm used to find the minimum
            string
        """
        print(title, ' :')
        print('the optimum weight vector is: ')
        print(self.sol)
        print('the obtained minimum square error is: ', self.min)
        return

    def plot_w_hat(self, algo_type=''):
        """ 
        plot_w_hat plots w_hat (solution of the minimization problem) 

        Parameters
        ----------
        title : typically the algorithm used to find w_hat, it is the title of the figure
            string
        """
        regressors = list(self.matr.columns)
        Nf = len(self.sol)
        nn = np.arange(Nf)
        plt.figure(figsize=(6, 4))
        plt.plot(nn, self.sol, '-o')
        ticks = nn
        plt.xticks(ticks, regressors, rotation=90)  # , **kwargs)
        plt.ylabel(r'$\^w(n)$')
        plt.title(f'{algo_type}-Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'./{algo_type}-what.png')
        plt.show()

    def plot_error_histagram(self, e, algo_type=''):
        plt.figure(figsize=(6, 4))
        plt.hist(e, bins=50, density=True, histtype='bar',
                 label=['training', 'test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(f'{algo_type}-Error histograms')
        plt.tight_layout()
        plt.savefig(f'./{algo_type}-hist.png')
        plt.show()

    def plot_regression_line(self, y_te, y_hat_te, algo_type=''):
        plt.figure(figsize=(6, 4))
        plt.plot(y_te, y_hat_te, '.')
        v = plt.axis()
        plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
        plt.xlabel(r'$y$')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title(f'{algo_type}-test')
        plt.tight_layout()
        plt.savefig(f'./{algo_type}-yhat_vs_y.png')
        plt.show()

    def print_statistics_of_errors(self, E_tr, y_tr, E_te, y_te):
        E_tr_mu = E_tr.mean()
        E_tr_sig = E_tr.std()
        E_tr_MSE = np.mean(E_tr ** 2)
        R2_tr = 1 - E_tr_sig ** 2 / np.mean(y_tr ** 2)
        E_te_mu = E_te.mean()
        E_te_sig = E_te.std()
        E_te_MSE = np.mean(E_te ** 2)
        R2_te = 1 - E_te_sig ** 2 / np.mean(y_te ** 2)
        rows = ['Training', 'test']
        cols = ['mean', 'std', 'MSE', 'R^2']
        p = np.array([[E_tr_mu, E_tr_sig, E_tr_MSE, R2_tr],
                      [E_te_mu, E_te_sig, E_te_MSE, R2_te]])
        results = pd.DataFrame(p, columns=cols, index=rows)
        return results


class SolveLLS(SolveMinProb):
    """ 
    Linear least squares: the optimum solution of the problem ||y-Aw||^2 is the
    w_hat=(A^TA)^{-1}A^Ty
    
    Inputs (inherited from class SolveMinProb)
    ------
    y : column vector with the regressand (Np rows)
        float
    A : Ndarray (Np rows Nf columns) 
    
    
    Methods
    -------
    run : runs the method and stores the optimum weights w in self.sol
            and the corresponding objective function in self.min
            
    Methods (inherited from class SolveMinProb)
    -------
    print_result : prints the optimum weights w_hat and the corresponding objective function
    plot_w_hat: plots the solution w_hat
            
    
    Attributes (inherited from class SolveMinProb)
    ----------
    matr : Ndarray A, with the measured data (Np rows, Nf columns)
        float
    vect : y, Ndarray of known values of the regressand (Np elements)
        float
    sol : w_hat, Ndarray with shape (Nf,), optimum value of vector w that minimizes ||y-Aw||^2
        float
    Np : number of available points (number of rows of A and y)
        integer
    Nf : number of available features (number of columns of A)
        integer
    min : value of ||y-Aw_hat||^2
        float
    err : not used, set to scalar 0.0
        float   
        
        
    Example: 
    -------
    lls=SolveLLS(y,A)    
    lls.run()    
    lls.print_result('Linear Least Squares')
    lls.plot_w_hat('Linear Least Squares')
        
    """

    def run(self):
        A = self.matr
        y = self.vect
        w_hat = np.linalg.inv(A.T @ A) @ (A.T @ y)
        self.sol = w_hat
        self.min = np.linalg.norm(A @ w_hat - y) ** 2
        self.err = 0
        return self.sol


class SolveMiniBatchGrad(SolveMinProb):

    def run(self, gamma=1e-3, Nit=100, num_batch=1):
        self.err = np.zeros((Nit, 2), dtype=float)
        self.gamma = gamma
        self.Nit = Nit
        A = self.matr
        y = self.vect

        w = np.random.rand(self.Nf, )  # random initialization of the weight vector
        sol_grad = np.zeros((self.Nf,), dtype=float)

        # iteration for finding optimal batch number
        size_batch = int(self.Np / num_batch)
        # mini batch gradient decent algoritm:
        for it in range(Nit):
            n_size = 0
            while n_size <= self.Np:
                # select mini batch and calculate gradient
                if n_size + size_batch <= self.Np:
                    # mini batch
                    # X_tr_norm[n_size:n_size+size_batch]
                    # y_tr_norm[n_size:n_size+size_batch]
                    grad = 2 * A[n_size:n_size + size_batch].T @ (A[n_size:n_size + size_batch] @ w -
                                                                  y[n_size:n_size + size_batch])
                    n_size = n_size + size_batch
                else:
                    # mini batch
                    # X_tr_norm[n_size:]
                    # y_tr_norm[n_size:]
                    grad = 2 * A[n_size:].T @ (A[n_size:] @ w - y[n_size:])
                    n_size = self.Np + 1
                w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(A @ w - y) ** 2
        self.sol = w
        min_grad = self.err[it, 1]
        return self.sol


class SolveStochGradWithAdam(SolveMinProb):

    def run(self, gamma=1e-3, Nit=100, beta1=0.9, beta2=0.999, e=1e-8):
        self.err = np.zeros((Nit, 2), dtype=float)
        self.gamma = gamma
        self.Nit = Nit
        A = self.matr
        y = self.vect

        w = np.random.rand(self.Nf, )  # random initialization of the weight vector
        sol_grad = np.zeros((self.Nf,), dtype=float)

        m = 0  # first momentum
        v = 0  # second momentum
        # stochastic gradient with adam algoritm:

        for it in range(Nit):
            n_size = 0
            for j in range(0, self.Np - 1):
                grad = 2 * A[j:j + 1].T @ (A[j:j + 1] @ w - y[j:j + 1])
                # ADAM
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** (it + 1))
                v_hat = v / (1 - beta2 ** (it + 1))
                update_grad = m_hat / (np.sqrt(v_hat) + e)
                w = w - gamma * update_grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(A @ w - y) ** 2
        self.sol = w
        min_grad = self.err[it, 1]
        return self.sol

