import numpy as np
from joblib import dump, load
import warnings

'''
This code is due to Dmitry Burov and developed for the paper Burov, Giannakis, Manohar and Stuart (2021).

D. Burov, D. Giannakis, K. Manohar and A. Stuart (2021), Kernel analog forecasting:
Multiscale test problems, Multiscale Modeling & Simulation 19(2), 1011–1040.

'''

class DATools:
  '''
  A simple class to provide basic tools for data assimilation with time-series
  serving as input.
  Convention for pairs is the following: they should come in a numpy matrix with
  vertical dimension equal to the total number of pairs and horizontal dimension
  equal to (dim(xk) + 1); that is, rows represent concatenated data.

  Methods:
    set_pairs     : set (xk, yk) pairs
    plot_scatter  : plot scatter plots in 1D
    learn_gpr     : learn Gaussian Process Regression using (xk, yk) pairs
    get_gpr       : return a function that predicts according to GPR
  '''

  def __init__(_s, mem_thrsh = 500):
    _s.pairs = None
    _s.pairs_sample = None
    _s.__p_set = False
    _s.__p_sample_set = False
    _s.mem_thrsh = mem_thrsh
    _s.gpr = None
    
  def set_gpr(_s, gp):
        _s.gpr = gp

  def set_pairs(_s, pairs):
    if pairs.ndim != 2:
      raise ValueError("'pairs' must be a 2D-array")
    _s.pairs = pairs
    _s.__p_set = True
    print("'pairs' set, # of samples: {}".format(pairs.shape[0]), flush=True)

  def get_mean_std(_s):
    range_min = np.min(_s.pairs[:,0])
    range_max = np.max(_s.pairs[:,0])
    mesh = np.linspace(range_min, range_max, 1000)
    mean, std = _s.gpr.predict(mesh[:,np.newaxis], return_std = True)
    return np.vstack( (mesh, mean, std) )

  def save_data(_s, index = None):
    mmstd = _s.get_mean_std()
    if index is None:
      np.save('gp_pairs.npy', _s.pairs)
      np.save('gp_pairs_sample.npy', _s.pairs_sample)
      np.save('gp_mean.npy', mmstd)
    else:
      np.save('gp_pairs_{}.npy'.format(index), _s.pairs)
      np.save('gp_pairs_sample_{}.npy'.format(index), _s.pairs_sample)
      np.save('gp_mean_{}.npy'.format(index), mmstd)
    return

  def plot_fit(_s):
    from matplotlib import pyplot
    import pdb; pdb.set_trace()
    if _s.pairs.shape[1] >= 3:
      print("Cannot plot mean and std for dim(xk) >= 2")
      return
    mmstd = _s.get_mean_std()
    mesh, mean, std = mmstd[0], mmstd[1], mmstd[2]
    
    fig, ax = pyplot.subplots(figsize=(16,10))
    ax.plot(mesh, mean, '.', ms = 4, color='black')
    ax.set_title('Function m', fontsize=30)
    fig.savefig('./figures/m.png')

  def plot_scatter(_s):
    from matplotlib import pyplot
    if _s.pairs.shape[1] >= 3:
      print("Cannot plot scatter plots for dim(xk) >= 2")
      return
    pyplot.figure()
    pyplot.plot(_s.pairs[:,0], _s.pairs[:,-1], '.',
        ms = 3, alpha = 0.5, color='gray')
    pyplot.plot(_s.pairs_sample[:,0], _s.pairs_sample[:,-1], '.',
        ms = 3, alpha = 0.8, color='red')
    _s.plot_fit()

  def plot_scatter2d(_s):
    from matplotlib import pyplot
    if _s.pairs.shape[1] not in {3, 4}:
      print("Cannot plot 2d scatter plots for dim(xk) not in {2,3}")
      return
    pyplot.figure()
    pyplot.scatter(_s.pairs[:,1], _s.pairs[:,0], c = _s.pairs[:,-1],
        cmap = 'magma')
    pyplot.colorbar()

    if _s.pairs.shape[1] == 4:
      pyplot.figure()
      pyplot.scatter(_s.pairs[:,1], _s.pairs[:,2], c = _s.pairs[:,-1],
          cmap = 'magma')
      pyplot.colorbar()

  def learn_gpr(_s, mem_thrsh = None, kernel = 'rbf'):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from time import perf_counter as timer

    if mem_thrsh is not None:
      _s.mem_thrsh = mem_thrsh
    n = _s.pairs.shape[0]
    if n > _s.mem_thrsh:
      inds = np.random.choice(n, size = _s.mem_thrsh, replace = False)
      n_sample = _s.mem_thrsh
    else:
      inds = np.s_[1:n]
      n_sample = n
    _s.pairs_sample = _s.pairs[inds]
    _s.__p_sample_set = True

    if kernel == 'matern':
      GP_ker = 1.0 * Matern(length_scale = 3, nu = 1.5)
    else: # including 'rbf', which is the default
      if kernel != 'rbf':
        warnings.warn(
            "Kernel '{}' is not supported; falling back to RBF".format(kernel)
        )
      #GP_ker = 1.0 * RBF(3, (1e-10, 1e+6)) + WhiteKernel()
      GP_ker = 1.0 * RBF(3, (1e-10, 1e+6))

    _s.gpr = GaussianProcessRegressor(
        kernel = GP_ker,
        n_restarts_optimizer = 15,
        alpha = 1
        #alpha = 0.5
    )

    start = timer()
    #LEARNING FROM PAIRS
    _s.gpr.fit(_s.pairs_sample[:,:-1], _s.pairs_sample[:,-1])
    #save gpr
    dump(_s.gpr, 'closure.joblib')
    
    elapsed = timer() - start

    print("GPR complete, time: {:.2f}".format(elapsed))
    print("GPR posterior kernel:", _s.gpr.kernel_, flush=True)

  def get_gpr(_s):
    return _s.gpr.predict
    #return _s.gpr.sample_y