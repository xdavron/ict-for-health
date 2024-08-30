# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sub.minimization as mymin
import matplotlib.pyplot as plt


if __name__ == '__main__':

    plt.close('all')  # close all the figures that might still be open from previous runs
    x = pd.read_csv("data/parkinsons_updrs.csv")  # read the dataset; xx is a dataframe
    x.describe().T  # gives the statistical description of the content of each column
    x.info()
    features = list(x.columns)
    print(features)
    # features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
    #       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    #       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    #       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    X = x.drop(['subject#', 'test_time'], axis=1)  # drop unwanted features
    Np, Nc = X.shape  # Np = number of rows/ptients Nf=number of features+1 (total UPDRS is included)
    features = list(X.columns)
    # %% correlation
    Xnorm = (X - X.mean()) / X.std()  # normalized data
    c = Xnorm.cov()  # note: xx.cov() gives the wrong result

    plt.figure()
    plt.matshow(np.abs(c.values), fignum=0)
    plt.xticks(np.arange(len(features)), features, rotation=90)
    plt.yticks(np.arange(len(features)), features, rotation=0)
    plt.colorbar()
    plt.title('Covariance matrix of the features')
    plt.tight_layout()
    plt.savefig('./corr_coeff.png')  # save the figure
    plt.show()

    plt.figure()
    c.motor_UPDRS.plot()
    plt.grid()
    plt.xticks(np.arange(len(features)), features, rotation=90)  # , **kwargs)
    plt.title('corr. coeff. among motor UPDRS and the other features')
    plt.tight_layout()
    plt.show()

    plt.figure()
    c.total_UPDRS.plot()
    plt.grid()
    plt.xticks(np.arange(len(features)), features, rotation=90)
    plt.title('corr. coeff. among total UPDRS and the other features')
    plt.tight_layout()
    plt.show()
    # %% Generate the shuffled data
    np.random.seed(273331)  # set the seed for random shuffling
    indexsh = np.arange(Np)
    np.random.shuffle(indexsh)
    Xsh = X.copy(deep=True)
    Xsh = Xsh.set_axis(indexsh, axis=0, inplace=False)
    Xsh = Xsh.sort_index(axis=0)
    # %% Generate training, validation and test matrices
    Ntr = int(Np * 0.75)  # number of training points
    Nte = Np - Ntr  # number of test points
    # %% evaluate mean and st.dev. for the training data only
    X_tr = Xsh[0:Ntr]  # dataframe that contains only the training data
    mm = X_tr.mean()  # mean (series)
    ss = X_tr.std()  # standard deviation (series)
    my = mm['total_UPDRS']  # mean of motor UPDRS
    sy = ss['total_UPDRS']  # st.dev of motor UPDRS
    # %% Normalize the three subsets
    Xsh_norm = (Xsh - mm) / ss  # normalized data
    ysh_norm = Xsh_norm['total_UPDRS']  # regressand only
    Xsh_norm = Xsh_norm.drop('total_UPDRS', axis=1)  # regressors only

    X_tr_norm = Xsh_norm[0:Ntr]  # X train
    X_te_norm = Xsh_norm[Ntr:]  # X test
    y_tr_norm = ysh_norm[0:Ntr]  # y train
    y_te_norm = ysh_norm[Ntr:]  # y test
    #%% use LLS to find w as the pseudo-inverse of A

    m = mymin.SolveLLS(y_tr_norm, X_tr_norm)  # instantiate the object that performs linear least squares
    w_hat = m.run()  # run LLS
    Nf = len(w_hat)
    y_hat_te_norm = X_te_norm @ w_hat
    MSE_norm = np.mean((y_hat_te_norm - y_te_norm) ** 2)
    MSE = sy ** 2 * MSE_norm

    # %% plots for LLS
    m.plot_w_hat('LLS')
    # plot the error histogram
    E_tr = (y_tr_norm - X_tr_norm @ w_hat) * sy  # training
    E_te = (y_te_norm - X_te_norm @ w_hat) * sy  # test
    e = [E_tr, E_te]

    m.plot_error_histagram(e, 'LLS')
    # plot the regression line
    y_hat_te = (X_te_norm @ w_hat) * sy + my
    y_te = y_te_norm * sy + my

    m.plot_regression_line(y_te, y_hat_te, 'LLS')

    # %% statistics of the errors
    y_tr = y_tr_norm * sy + my
    y_te = y_te_norm * sy + my
    results = m.print_statistics_of_errors(E_tr, y_tr, E_te, y_te)
    print('LLS results')
    print(results)
    # %% Mini Batch Gradient algorithm
    gamma = 1e-5
    Nit = 200
    num_batch = 10

    # for j in range(50,Nit,50):
    #     for i in range(1, num_batch, 4):
    g = mymin.SolveMiniBatchGrad(y_tr_norm, X_tr_norm)
    sol_grad = g.run(gamma, Nit, num_batch)
    # MSE calculation
    y_hat_te_norm_grad = X_te_norm @ sol_grad
    MSE_norm_grad = np.mean((y_hat_te_norm_grad - y_te_norm) ** 2)
    MSE_grad = sy ** 2 * MSE_norm_grad
    print(f'iter: {Nit} batch: {num_batch} MSE: {MSE_grad}')

    # %% plots mini batch
    # plot the optimum weight vector
    g.plot_w_hat('Mini Batch')
    # plot the regression line
    y_hat_te_minigrad = (X_te_norm @ sol_grad) * sy + my
    y_te = y_te_norm * sy + my
    g.plot_regression_line(y_te, y_hat_te_minigrad, 'Mini Batch')
    # plot the error histogram
    E_tr_mini = (y_tr_norm - X_tr_norm @ sol_grad) * sy  # training
    E_te_mini = (y_te_norm - X_te_norm @ sol_grad) * sy  # test
    e = [E_tr_mini, E_te_mini]
    g.plot_error_histagram(e, 'Mini Batch')
    g.plot_err('Mini Batch')
    # %% statistics of the errors MiniBatch
    y_tr = y_tr_norm * sy + my
    y_te = y_te_norm * sy + my
    results = g.print_statistics_of_errors(E_tr_mini, y_tr, E_te_mini, y_te)

    print('Mini Batch results')
    print(results)

    # %% Stochastic Gradient algorithm
    gamma = 1e-3
    Nit = 30
    beta1 = 0.9
    beta2 = 0.999
    e = 1e-8

    p = mymin.SolveStochGradWithAdam(y_tr_norm, X_tr_norm)
    sol_stoch = p.run(gamma, Nit, beta1, beta2, e)
    # MSE calculation
    y_hat_te_norm_stoch = X_te_norm @ sol_stoch
    MSE_norm_stoch = np.mean((y_hat_te_norm_stoch - y_te_norm) ** 2)
    MSE_stoch = sy ** 2 * MSE_norm_stoch
    # %% E{(y−xˆwL)^2}for training datasets as a function of the iteration step i
    # plot
    # print(err_stoc[:,0])
    # p.plot_err('SGD with Adam')
    # %% plots stochastic
    p.plot_w_hat('SGD with Adam')
    # plot the regression line
    y_hat_te_stoch = (X_te_norm @ sol_stoch) * sy + my
    y_te = y_te_norm * sy + my
    p.plot_regression_line(y_te, y_hat_te, 'SGD with Adam')
    # plot the error histogram
    E_tr_stoch = (y_tr_norm - X_tr_norm @ sol_stoch) * sy  # training
    E_te_stoch = (y_te_norm - X_te_norm @ sol_stoch) * sy  # test
    e = [E_tr_stoch, E_te_stoch]
    p.plot_error_histagram(e, 'SGD with Adam')
    p.plot_err('SGD with Adam')
    # %% statistics of the errors Stochastic
    y_tr = y_tr_norm * sy + my
    y_te = y_te_norm * sy + my
    results = p.print_statistics_of_errors(E_tr_stoch, y_tr, E_te_stoch, y_te)
    print('Stochastic results')
    print(results)

