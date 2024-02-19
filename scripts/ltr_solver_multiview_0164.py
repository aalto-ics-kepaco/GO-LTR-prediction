######################
## Version 0.16.4 #######
######################
"""
**********************************************************************
   Copyright(C) 2020-2023 Sandor Szedmak  
   email: sandor.szedmak@aalto.fi
          szedmak777@gmail.com

   This file contains the code for Polynomial regression via latent
   tensor reconstruction (PRLTR.

    MIT License

    Copyright (c) 2023 KEPACO

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

***********************************************************************/
"""

"""
Polynomial regression via latent tensor reconstruction
Multiview version 

Version 0.16.4 (16.07.2023)
"""

"""
--- Vector output, one view, ranks range processed as in one step
polynomial encoding, a(X\lambda_{u}U)V^{T} 
--- multiview learning is implemented
--- Output notmalization, mean, and sclaing by  L_infty are included
--- Input normalization,  mean, and scaling by L_2 and(or) L_infty are included
--- Quantile regression is added
--- Projection into a ball instead of onto.
--- Multitask and federated learning extensions
----- augmented gradient: local gradient + global adjusment
----- start without parameter and gradient initialization
--- yindex is introduced for output table
--- normalization revised
--- data control on parameter operations, normalizaqtion, initialization, update    
--- external activation function is included
--- different activation function for each view
--- normalization of the xlambda for implicit function learning
--- view wise rankuv
--  sparse input arrays
--- export latent views
--- external activation function + derivative
--- externel loss function + gradient
--- mini-batch metropolis-hasting sampling
--- mini-batch sampling burn-in delay
--- loss_function_diff function has additional parameter (cmodel = None(self))
"""

## #####################################################
import sys
import time
import pickle

import numpy as np
import scipy as sp

## #########################################
## gpu setting
torch = None
# try:
#   import torch
# except ImportError: 
#   torch = None

gc = None

## ################################################################
## ################################################################
## Helper classes
## ################################################################
class store_cls:

  def __init__(self):
    self.lPstore = None
    return

## ################################################################
class tensor_param_cls:
  """
  Task: to store a parameter tensor, gradients, and provide initialization,
        and update functions
  """
  def __init__(self, tshape, norm_axis=None):
    """
    Task:   to set the tensor and related parameters
    Input:  tshape      tuple of tensor shape
            norm_axis   axis of L2 norm normalization
            struct     =1  tshape defines list of arrays, tshape[0] is list size
                            tshape[1:] gives the array shape
                        =0  tshape defines array
    """

    self.tshape = tshape     ## shape  
    self.norm_axis = norm_axis  ## axis of normalization

    self.xT = None           ## tensor parameter 
    self.xTnext = None       ## tensor parameter Nesterov accelerated grad offset
    
    ## gradient related variables
    self.xTGrad = None       ## tensor gradient
    self.xATGrad = None      ## aggregated(moment) gradient
    self.xnATGrad = None     ## aggregated gradient norm

    ## global gradient to modify the local one
    self.xTGradglobal = None  ## the global gradient  

    return
  
  ## --------------------------------------
  def init_tensor(self,init_mode=0, rng=None, rng_seed=None):
    """
    Task: to initialize the tensor
    Input init_mode     =0 zero, =1 ones, =2 random normal
          rng           reference of random generator               
    Modify: self.xT
    """
    if rng is None:
      rngin = np.random.default_rng(rng_seed) 
    else:  
      rngin = rng

    if init_mode == 0:       ## zero
      self.xT = np.zeros(self.tshape)
    elif init_mode == 1:     ## ones
      self.xT = np.ones(self.tshape)
    elif init_mode == 2:     ## random normal
      self.xT = rngin.standard_normal(size = self.tshape)
    elif init_mode == 3:   ## uniform 2*() -1
      self.xT = 2*sp.stats.uniform.rvs(size = self.tshape) -1
    elif init_mode == 4:   ## cumulative  
      alpha = 2
      beta = 0
      self.xT = sp.stats.levy_stable.rvs(alpha,beta, loc = 0, scale = 1, \
                                    size = self.tshape, random_state = None)
      self.xT = np.cumsum(self.xT,axis = 1)
      m,n = self.xT.shape
      sn = np.sqrt(np.arange(n)+1)
      self.xT /= sn

    return

  ## --------------------------------------
  def init_tensor_gradient(self):

    tdim = len(self.tshape)
    self.xTGrad = np.zeros(self.tshape)
    self.xATGrad = np.zeros(self.tshape)

    if tdim > 1:
      tshape_reduced = []
      for i in range(tdim):
        if i != self.norm_axis:
          tshape_reduced.append(self.tshape[i])
      self.xnATGrad = np.zeros(tshape_reduced)
    else:
      self.xnATGrad = 0
    
    self.xTnext = np.zeros(self.tshape)

    return

  ## ----------------------------------
  def normalize(self, iscale_axis=None, xrescale=None, inorm_scale=None, \
                  radius=1, norm_type=0, rng=None):
    """
    Task to notmalize the tensor along the norm axis
    Input:  iscale_axis   axis on which the pruduct of norms computed
            xrescale      tensor to scale the tensor parameter before normalization
            inorm_scale   rescale the norm along the normalization axis
            radius        radius of the ball in normalization, =1 unit ball
            norm_type     =0 L2 norm, 
                          =1 projection of hyperboloid model into Poincare disc
    output: normlambda    product of the norm to propagate for other parameters
    """

    tdim=len(self.tshape)
    if radius == 0 or radius is None:
      radius = 1
      
    if xrescale is not None:
      self.xT *= xrescale

    if self.norm_axis is not None:
      if norm_type == 0:
        xnorm = np.sqrt(np.sum(self.xT**2,self.norm_axis))
      elif norm_type == 1:
        xnorm = np.sum(np.abs(self.xT),self.norm_axis)
      elif norm_type == 2:
        xnorm = np.max(np.abs(self.xT),self.norm_axis)
      elif norm_type == 3:
        xT = np.arcsinh(self.xT)
        xnorm = np.sqrt(np.sum(xT**2,self.norm_axis))
      elif norm_type == 4:
        xT = np.tanh(self.xT)
        xnorm = np.sqrt(np.sum(xT**2,self.norm_axis))
      elif norm_type == 5:
        xT = self.xT*(self.xT > 0)   ## RELU
        xnorm = np.sqrt(np.sum(xT**2,self.norm_axis))

      xnorm /= radius
      xnorm = xnorm + 1*(xnorm == 0)
      
    if inorm_scale == 1:
      xnorm /= (self.tshape[self.norm_axis]**0.2)

    if self.norm_axis is not None:
      if tdim>1:
        ## move normalization axis to the front
        self.xT = np.moveaxis(self.xT,self.norm_axis,0)  
        self.xT /= xnorm     ## exploit broadcasitng on the last dims
        self.xT = np.moveaxis(self.xT,0,self.norm_axis)  ## move norm axis back 

      else:
        self.xT /= xnorm

    if iscale_axis is not None:  
      if len(xnorm.shape)>1:
        xnormlambda = np.prod(xnorm,iscale_axis)  ## to scale for lambda or lambdaU
      else:
        xnormlambda = xnorm
    else:
      xnormlambda = 1

    return(xnormlambda)
  ## ---------------------------------
  def nag_offset(self, gammaoff=0, psign=-1):
    """
    Input: gammaoff =0 no change , gradient scale otherwise
           psign    =0 no change, -1 pull back, +1 push forward
    """

    pgamma = gammaoff * psign
    self.xTnext = self.xT + pgamma*self.xATGrad

    return

  ## ---------------------------------
  def adam_update(self, opt_params=None):
    """
    Task: to implement the ADAM gradient update
    Input: opt_params     reference to object containing the optimization parameters
    Modify: self.xTGrad, self.xATGrad, self.xnATGrad
    """

    if opt_params is None:
      return

    tdim=len(self.tshape)

    iproject  = 0
    if iproject == 1 and self.norm_axis is None:    
      ## gradient projection to the tangent plane
      self.xTGrad -= (np.sum(self.xT*self.xTGrad,self.norm_axis) \
                        /np.sum(self.xT*self.xT,self.norm_axis))*self.xT 

    ## perturbation
    if opt_params.perturb > 0:
      if opt_params.rng is None:
        rng = np.random.default_rng()
      else:
        rng = opt_params.rng
      xGperturb = rng.standard_normal(size = self.tshape)
      nxGperturb = np.sqrt(np.sum(xGperturb**2))
      nxG=np.sqrt(np.sum(self.xTGrad**2))
      self.xTGrad += opt_params.perturb * nxG * xGperturb/nxGperturb

    ## Global gradient adjustment
    if self.xTGradglobal is not None:
      self.xTGrad += opt_params.sigmaglobal * self.xTGradglobal

    ## norm on the normalization axis
    if self.norm_axis is not None:
      nTgrad = np.sum((opt_params.sigmacorrect*self.xTGrad)**2,self.norm_axis) 
    else:
      nTgrad = np.sum((opt_params.sigmacorrect*self.xTGrad)**2) 

    self.xATGrad = opt_params.gammanag*self.xATGrad+(1-opt_params.gammanag) \
      *opt_params.sigmacorrect*self.xTGrad  
    vhat = self.xATGrad/(1-opt_params.gammanag**opt_params.iter)

    self.xnATGrad = opt_params.gammanag2*self.xnATGrad+(1-opt_params.gammanag2)*nTgrad
    ngradhat = self.xnATGrad/(1-opt_params.gammanag2**opt_params.iter)

    ## denominator !!!
    xdenom = np.sqrt(ngradhat) + opt_params.ngeps
    if self.norm_axis is not None:
      if tdim > 1: 
        ## move normalization axis to the front
        vhat = np.moveaxis(vhat,self.norm_axis,0)
        vhat /= xdenom
        vhat = np.moveaxis(vhat,0,self.norm_axis)  ## move the front axis back 
        self.xT = opt_params.gamma*self.xT \
            -opt_params.sigmabase*opt_params.sigma * vhat
      else:
        self.xT = opt_params.gamma*self.xT \
          -opt_params.sigmabase*opt_params.sigma * vhat / xdenom 
    else:
      self.xT = opt_params.gamma*self.xT \
          -opt_params.sigmabase*opt_params.sigma * vhat / xdenom 
 
    return

## ################################################################
def setcuda():

  ## torch = None
  cudadevice = None
  if torch is not None:
    if  torch.cuda.is_available() is True:
      cudadevice=torch.device('cuda')

  return(cudadevice)

## ################################################################
## ######## MAIN CLASS  
## ################################################################
class ltr_solver_cls:
  """
  Task: to implement the latent tensor reconstruction based
        polynomial regression 
  """
  
  ## -------------------------------------------------
  def __init__(self, norder=1, rank=10, rankuv=None, \
               loss_function = None, loss_function_diff = None, \
               activation_function = None, activation_function_diff = None):
    """
    Task: to initialize the ltr object
    Input:  norder    input degree(order) of the polynomial
            nordery   output degree, in the current version it has to be 1 !!!
            rank      the number of rank-one layers
            rankuv    common rank in UV^{T}  U ndimx,rankuv, V nrank,nrankuv
    """
    global gc   ## numerical package address = np or torch

    self.ifirstrun = 1       ## =1 first fit run
                           ## =0 next run for new data or 

    self.norder = norder      ## input degree(order)
    ## -----------------------------------------
    self.nrank0 = rank   ## initial saved maximum rank
    ## initial maximum rank, in the rank extension self.nrank it is changed
    ## if in function fit irank_add=1 
    self.nrank=rank
    ## ------------------------------------------
    if rankuv is not None:
      self.nrankuv = rankuv  ## P=UV^{T},  U ndimx,nrankuv  V nrank,nrankuv
    else:
      self.nrankuv = self.nrank0

    self.lrankuv = None      ## rankuv for all views set by fit()  

    ## set backends, torch, cuda
    self.ibackend = 0   ## =0 numpy =1 torch
    if self.ibackend == 0:
      gc = np
    elif self.ibackend == 1:
      if torch is not None:
        gc = torch
      else:
        gc = np
    
    self.cudadevice = None
    if self.ibackend == 1 and torch is not None:
      self.cudadevice=setcuda()

    self.ndimx = 0        ## vinput dimensions
    self.xinputdimx = None  ## vector of number of columns in the inputs
    self.xviewdimx = None   ## vector of number of columns in the views
    self.ninput = 0       ## number of input matrices
    self.nview = 0        ## number of views
    self.ndimy = 0        ## output dimension

    self.nrepeat = 10     ## number of epochs 
    self.iter = 0         ## iteration counter used in the ADAM update
    self.icount = 0       ## number of processed minibatch
    
    self.cregular = 0.000005
    self.clambda = self.cregular      ## lambda regularization constant

    ## parameter objects
    self.xU = None      ## input poly parameters (order,rankuv,ndimx)
    self.xV = None      ## input poly parameters (order,rank,nrankuv)
    self.xQ = None      ## output poly parameter  (rank,dimy)
    self.xlambda = None     ## lambda factors 
    self.xlambdaU = None    ## lambda factors 
    self.lParams = []       ## list of all params
    self.lParamshapes = []  ## list of parameter shape tuples

    self.ilambda = 1         ## xlambda is updated by gradient
    self.ilambda_norm = 0    ## xlambda is normalized aftre update

    ## self.f = None           ## computed function value
    self.iecum = 0        ## =1 discounting of erro =0 no 
    self.Ecum = None      ## the accumulated, discounted error value
    self.Ecum0 = None     ## the accumulated, discounted initial error value
    self.ediscount0 = 1   ## discount factor starting value 
    self.ediscount = self.ediscount0  ## discount factor 

    self.irandom = 1  ## =1 random block =0 order preserving blocks
    self.iscale = 1   ## =1 error average =0 no

    ## ADAM + NAG
    self.sigma0 = 0.05     ## initial learning speed
    self.sigma = 0.05     ## updated learning speed
    self.sigmabase = 1     ## sigma scale
    self.gamma = 1.0     ## discount factor
    self.gammanag = 0.95  ## Nesterov accelerated gradient factor
    self.gammanag2 = 0.95  ## Nesterov accelerated gradient factor
    self.ngeps = 0.00001     ## ADAM correction to avoid 0 division 
    self.nsigma = 10 ## range without sigma update
    self.dscale = 2   ## x-1/dscale *x^2 stepsize update
    self.sigmamax = 1   ## maximum sigma*len_grad
    self.sigmacorrect = 1   ## gradient correction if it is too long
    self.perturb = 0.0    ## gradient perturbation
    ## learning speed for global gradient adjustment
    self.sigmaglobal = self.sigma0  ## learning speed for global gradient adjustment
    
    ## implicit normalization and homogeneous coordinates
    self.iymean = 1      ## =1 output vectors are averaged
    self.ymean = 0        ## ymean
    self.iyscale = 1      ## =1 output vectors are scaled
    self.yscale = 1       ## the scaling value

    self.ixmean = 0     ## =1 input views centralized to training mean
    self.lxmean = []    ## list ov view means
    self.ixscale = 1    ## =1 input views are scaled by L_infty norm to 1
    self.lxscale = []      ## the l_infty scale values
    self.ixl2norm = 0    ## =1 input views are scaled by L_2 norm to 1
    self.lxl2norm = []   ## the L_2 scale values

    self.ihomogeneous = 1  ## =1 input views are homogenised =0 not 

    ## mini batches
    self.mblock = 100          ## data block size, number of examples
    self.mblock_gap = None    ## shift of blocks
    self.ntestblock = 5     ## test block size = ntestblock * mblock

    self.ibias = 1          ## =1 bias is computed =0 otherwise
    self.pbias = None       ## bias vector with size ndimy

    self.inormalize = 1     ## force normalization in each iteration
    self.radius = 1.0      ## radius of the L2 ball in normalization
    self.norm_type = 0    ## parameter normalization =0 L2 norm
                          ## see other options in tensor_param_cls.normalize()

    ## parameter variables which can be stored after training
    ## and reloaded in test
    self.lsave = ['norder','nrank','iyscale','yscale','ndimx','ndimy', \
                'xU','xV','xQ','xlambda','xlambdaU','pbias']
    ## test environment
    self.istore = 0
    self.cstore_numerator = store_cls()
    self.cstore_denominator = store_cls()
    self.cstore_output = store_cls()
    self.store_bias = None
    self.store_yscale = None
    self.store_lambda = None
    self.store_grad = []
    self.store_acc = []

    self.max_grad_norm = 0  ## maximum gradient norm 
    self.maxngrad = 0       ## to signal of long gradients

    self.Ytrain = None        ## deflated output vectors

    ## activation function
    self.iactfunc = 0         ## =0 identity, =1 arcsinh =2 2*sigmoid-1 =3 tanh =4 relu
    self.iactfunc_ext = 0     ## =0 identity, =1 arcsinh =2 2*sigmoid-1 =3 tanh =4 relu
    self.lactfunc = None      ## activation function for each view
    self.lactfunc_ext = None  ## external activation function for each view

    ## external or internal activations
    if activation_function is None:
      self.activation_function = self.activation_func
    else:
      self.activation_function = activation_function

    if activation_function_diff is None:
      self.activation_function_diff = self.activation_func_diff
    else:
      self.activation_function_diff = activation_function_diff

    ## external or internal loss
    if loss_function is None:  
      self.loss_function = self.loss_func
    else:  
      self.loss_function = loss_function
      
    if loss_function_diff is None:  
      self.loss_function_diff = None
    else:  
      self.loss_function_diff = loss_function_diff
      
    ## loss degree
    ## old version  
    self.lossdegree = 0  ## =0 L_2^2, =1 L_2^1, =0.5 L_2^1.5
    ## new version
    self.lossdegree_internal = - self.lossdegree + 2   
    
    ## power regularization
    self.regdegree = 1   ## degree of regularization, default: Lasso

    ## quantile regression with L1 norm regression as subcase if quantile_alpha = 0.5
    self.iquantile = 0           ## norm based regression, =1 quantile regression
    self.quantile_alpha = 0.5   ## quantile, confidence, parameter
    self.quantile_smooth = 1    ## smoothing parameter of the pinball loss
                                 ## in exp case if it is larger then 
                                 ## it is closer to the pinball, but less smooth,
                                 ## in case of hyperbole smaller the t closer to pinball 
    self.quantile_scale = 1      ## scaling the gradient                            
    self.iquantile_hyperbola = 1  ## =0 logistic = 1 hyperbolic approximation

    self.report_freq = 100 ## state report frequency relative to the number of minibatches.

    ## random state 
    self.rng_seed = 12345
    self.rng = np.random.default_rng(12345)
    ## self.rng = None

    ## reweighting the data items
    self.idweight = 0         ## =0 no data weight =1 data reweighting
    self.dweight = None       ## weight of the training examples
    self.dweight_scale = 1  ## scaling the error term
    self.dweight_max = 10     ## max weight

    ## storing the complete design
    self.llinks = None

    ## learning implicit function
    self.implicit = 0       ## =0 explicit function f(x)->y , =1 F(y,x) = 0
    if self.implicit == 1:
      self.ilambda_norm = 1  ## in implicit function learnin lambda
                             ## is normalized

    ## mark thet the fit function is applied
    self.fitfinished = 0  ## =1 is set at the end of the fit()  

    ## sampling of blocks(mini_batches)
    self.imb_sample = 0   ## =1 sampling mini batches, =0 not
    self.mb_sample_delay = 5     ## number of iteration before
                                  ## starting the mini_batch sampling
    self.loss_previous = None        ## stored previous loss, first time None
    self.lviews_previous = None   ## stored previous input views
    self.y_previous = None        ## stored previous output 
    self.ibatch_previous = None   ## stored previous batch indexes
    self.fvalue = None            ## stored function value
    
    return

  ## ------------------------------------------------
  ## torch numpy diffrences
  def tensor_load(self,X):
    """
    Task: to call tensor loading into gpu if it is available in torch 
          np to torch if torch is availbale 
    """
    
    if self.ibackend == 1:
      if self.cudadevice is not None:
        xTor = gc.tensor(X,dtype = gc.float64,device = self.cudadevice)
      else:
        xTor = torch.from_numpy(X)
    else:  ## numpy
      xTor = X

    return(xTor)

  ## ------------------------------------------------
  def tensor2np(self,xT):
    """
    Task if xT torch tensor convert into numpy
    """
    if self.ibackend == 1:
      if self.cudadevice is not None:
        X = xT.cpu().data.numpy()
      else:
        X = xT.data.numpy()
    else:
      X = xT  ## numpy 
    
    return(X)

  ## ------------------------------------------------
  def compute_output_scaling(self,Y):
    """
    Task: to compute scale and centralization parameters
    Input: Y   2d array of output
    Modifies: self.lxmean, self.lxscale
    """

    ## process input matrices 
    if self.iymean == 1:
      self.ymean = np.mean(Y,0)
    
    if self.iyscale == 1:
      xnorm = np.max(np.abs(Y), 0)
      if sp.sparse.isspmatrix_coo(xnorm):
        xnorm = xnorm.toarray()[0]
      xnorm = xnorm + (xnorm == 0)
      self.yscale = xnorm

    return

  ## ------------------------------------------------
  def output_scaling(self,lY):
    """
    Task: to scale and centralize output
    Input: lY   list of output array [y]
    Output: list elements are modified
    """

    if self.iymean == 1:
      ny = len(lY)
      for i in range(ny):
        lY[i] -= self.ymean

    if self.iyscale == 1:
      ny = len(lY)
      for i in range(ny):
        lY[i] = np.true_divide(lY[i], self.yscale, out=lY[i], casting="unsafe")
        #lY[i] /= self.yscale
    
    return
  ## ------------------------------------------------
  def output_rescaling(self,lY):
    """
    Task: to rescale and centralize output
    Input: lY   list of output array [y]
    Output: list elements are modified
    """

    ## the order of scaling and centralization is reversed in the rescaling
 
    if self.iyscale == 1:
      ny = len(lY)
      for i in range(ny):
        lY[i] *= self.yscale
    
    if self.iymean == 1:
      ny = len(lY)
      for i in range(ny):
        lY[i] += self.ymean

    return

  ## ------------------------------------------------
  def input_homogeneous(self,lX):
    """
    Task: to add homogeneous coordinate to input views
    Input: lX   list of output array 
    Output: list elements are modified
    """

    if self.ihomogeneous == 1:
      nX = len(lX)
      for i in range(nX):
        if lX[i] is not None:
          m,n = lX[i].shape
          lX[i] = np.concatenate((lX[i], np.ones((m,1))), axis = 1)
    
    return

  ## ------------------------------------------------
  def compute_input_scaling(self,lX):
    """
    Task: to compute scale and centralization parameters
    Input: lX   list of arrays array [X,...,X]
    Output: list elements are modified
    Modifies: self.lxmean, self.lxscale
    """

    e1 = np.array([1])  ## for homogeneous coordinates

    nX = len(lX)
    ## process input matrices 
    if self.ixmean == 1:
      self.lxmean = [np.mean(lX[i],0) for i in range(nX) ]
    
    if self.ixscale == 1:
      self.lxscale = [ None for _ in range(nX)]
      for i in range(nX):
        xnorm = np.max(np.abs(lX[i]), 0)
        if sp.sparse.isspmatrix_coo(xnorm):  
          xnorm = xnorm.toarray()[0]
        xnorm = xnorm + (xnorm == 0)
        self.lxscale[i] = xnorm

    if self.ixl2norm == 1:
      self.lxl2norm = [ None for _ in range(nX)]
      for i in range(nX):
        if sp.sparse.isspmatrix_csr(lX[i]):  
          xnorm = np.sqrt(np.sum(lX[i].multiply(lX[i]), 1))
        else:
          xnorm = np.sqrt(np.sum(lX[i]**2, 1))
        xnorm = xnorm + (xnorm == 0)
        self.lxl2norm[i] = xnorm

    return

  ## ------------------------------------------------
  def input_scaling(self, lX, iblock=None):
    """
    Task: to scale and centralize output
    Input: lX   list of arrays array [X,...,X]
           iblock  vector of row indexes in the input arrays
    Output: list elements are modified
    """

    nX = len(lX)

    if self.ixmean == 1:
      for i in range(nX):
        if lX[i] is not None:
          lX[i] -= self.lxmean[i]
    
    if self.ixscale == 1:
      for i in range(nX):
        if lX[i] is not None:
          lX[i] /= self.lxscale[i]

    if self.ixl2norm == 1:
      for i in range(nX):
        if lX[i] is not None:
          lX[i] = (lX[i].T/self.lxl2norm[i][iblock[i]]).T

    return

  ## ------------------------------------------------
  def links_item_completion(self, litem, nX):
    """
    Task: to make the llinks items complete:
          - convert number into list
          - add degree 1 if it not degree is given
    Input:  litem  list or number corresponding an llinks item
            nX     number of arrays
    Output: litem  list of list [ [array 1 index, degree],
                   [array 2 index, degree],...]
    """

    ## check the indexes
    if not isinstance(litem,list): ## is it list
      if isinstance(litem,int):    ## is it integer
        litem = [[litem,1]]         ## index and degree               
      else:
        litem = [[0,1]]  ## if not list or integer then first array is chosen

    ## check intermediate lists
    nitem = len(litem)
    for i in range(nitem):   ## intermediate list
      if not isinstance(litem[i],list):  ## is it list
        if isinstance(litem[i],int):     ## is it integer
          ## if array index is out of lX then the first is used
          if litem[i] >= nX or litem[i] < 0:  
            litem[i] = 0
          litem[i] = [litem[i],1]           ## index and degree
      else:
        ## if array index is out of lX then the first is used
        if litem[i][0] >= nX or litem[i][0] < 0:
          litem[i][0] = 0
        if len(litem[i])>=2:
          if litem[i][1] < 1 or litem[i][1]>10:
            litem[i][1] = 1   ## to avoid negative or too high degree
        elif len(litem[i]) == 1:
          litem[i].append(1)   ## add degree
        else:
          litem[i] = [0,1]     ## add index and degree

    return(litem)

  ## ------------------------------------------------
  def minibatch_slice(self, lX, llinks, ibatchindex, xindex=None, lsubset = None):
    """
    Task: to collect the mini-batch slices for each input array
    Input: lX       list of input arrays
           llinks   list of views as joint input arrays
           ibatchindex  vector of mini-batch index renage
           xindex   2d array of row indexes in the input arrays,
                    columns for views
           lsubset  sub list of views in implicit function learning
    Output: lbatches  list of slices of the input arrays corresponding
                      to the mini-batch
            lblocks   the row indexes corresponding to the mini-batch
    """
    
    ninput = len(lX)
    nview = len(llinks) ## number of views
    lbatches = [ None for _ in range(ninput)]
    lblocks = [ None for _ in range(ninput)]

    if lsubset is None:
      lsubset = [ i for i in range(nview)]

    for iview in range(nview):
      if iview in lsubset:
        if xindex is None:
          iblock = ibatchindex
        else:
          iblock = xindex[ibatchindex,iview]
        litem = llinks[iview]
        nitem = len(litem)
        for i in range(nitem):
          itable = litem[i][0]
          if isinstance(lX[itable],np.ndarray):
            lbatches[itable] = lX[itable][iblock]
          elif sp.sparse.isspmatrix_csr(lX[itable]):
            lbatches[itable] = lX[itable][iblock].toarray()
          lblocks[itable] = iblock

    return(lbatches,lblocks)

  ## ------------------------------------------------
  def process_view_list(self, lX, llinks, iconcat=1, lsubset=None):
    """
    Task: to process the view lists, and to concatenate the arrays in the inner lists
          In the concatenation the horizontal case is applied, 
          if the first dimensions are different then the minimum of them is used, 
          thus the arrays are cut into that size.   
    Input: lXtrain       list of 2d arrays of training input data 
                         corresponding to the views, 
                         one view might have more than one data array
           llink         list of lists of lists, 
                         the most outer list, 0, enumerate the views,
                         in the intermediate list, 1, the index or list of
                         data arrays 
                         corresponding to one view are listed. 
                         in the most inner, 2, [ index , degree of the
                         data array]    
           iconcat       =1 concatenate arrays, =0 compute the number
                         of variables only
           lsubset       subset of views for implicit function
    Output: lXviews      list of arrays, the view arrays are the
                         concatenated ones
    """

    nview = len(llinks)
    lXviews = [ None for _ in range(nview)]
    if self.xviewdimx is None:
      self.xviewdimx = np.array([ 0 for _ in range(nview)])

    if lsubset is None:
      lsubset = [ i for i in range(nview)] ## all views

    for iview in range(nview):
      
      litem = llinks[iview]
      nitem = len(litem)
          
      if iview in lsubset:
        if nitem == 1:
          if iconcat == 1:
            lXviews[iview] = lX[litem[0][0]]**litem[0][1]
          nvar = lX[litem[0][0]].shape[1]    
        else:
          if iconcat == 1:
            lconcat = [ lX[litem[i][0]]**litem[i][1] for i in range(nitem)]
            lXviews[iview] = np.hstack(lconcat)
          nvar = np.sum( np.array([lX[litem[i][0]].shape[1] for i in range(nitem)]))

        if self.ihomogeneous == 1:
          self.xviewdimx[iview] = nvar + 1
        else:
          self.xviewdimx[iview] = nvar

    return(lXviews) 

  ## --------------------------------------------------
  def preprocess_minibatch(self, lXtrain, llinks, ibatchindex, xindex, lsubset = None):
    """
    Task: to compile the data block corresponding to the  the minibatch 
    Input:  lXtrain      list of input table arrays
            llinks       list of designs
            ibatchindex  vector of indexes contained in the minibatch
            xindex       joining array data tables
            lsubset      list of subset of views for implicit function
    Output: lx_batch     list 2d arrays of slices of the data views 
                         correponding to the minibatch
    """
        
    ## load the mini-batches
    lblocks,lsliceindex = self.minibatch_slice(lXtrain, llinks, ibatchindex, \
                          xindex, lsubset = lsubset)
    if self.ixl2norm == 1:
      ## mean and scaling by l2 norm of rows
      self.input_scaling(lblocks,iblock = lsliceindex)  
    else:
      ## mean and scaling l_infty norm of columns
      self.input_scaling(lblocks)  

    lx_batch = self.process_view_list(lblocks,llinks, lsubset = lsubset)
    self.input_homogeneous(lx_batch)  ## add homogeneous coordinates

    return(lx_batch)

  ## ------------------------------------------------
  def param_array(self, iview=0):
    """
    Task: to return the complete parameter array for a view
          d = iview
          P^{d) = V^{d}(1_{rankuv}\lambdaU^{(d)}^{T} \circ U^{d}T)
    Input:  iview   view index 
    Output  Pd      complete parameter array of view dview
                    Pd.shape = self.nrank, self.xviewdimx[iview]  

    """
    ## if the index is out of view range then the first view is selected 
    if iview < 0 or iview>= self.norder:
      iview = 0

    xU = self.xU[iview].xT
    xV = self.xV[iview].xT
    xlambdaU = self.xlambdaU[iview].xT

    Ulambda = xU * np.outer(xlambdaU,np.ones(self.lrankuv[iview]))
    Pd = np.dot(Ulambda,xV.T)   

    return(Pd)
 
  ## ------------------------------------------------
  def init_tensorparams(self, init_mode=2):
    """
    Task: to initialize the parameter vectors U,V,Q, bias,lambda lambdaU
          xU,xV,xLambdaU are lists of size norder
    Input:
    Output:
    Modifies: self.xU, self.xV, self,xQ, self.xlambda, self.xlambdaU, \
              self.pbias
    """
  
    nrank = self.nrank
    norder = self.norder
    lrankuv = self.lrankuv
    ndimy = self.ndimy

    if self.rng is None:
      rng = np.random.default_rng(self.rng_seed)
    else:
      rng = self.rng

    ## tshape=(self.xviewdimx[d],lrankuv[d])
    self.xU = [tensor_param_cls((self.xviewdimx[d],lrankuv[d]),norm_axis=1) \
                 for d in range(norder)]
    for d in range(norder):
      self.xU[d].init_tensor(init_mode = init_mode, rng = rng)

    self.xV = [ tensor_param_cls((nrank,lrankuv[d]),norm_axis=1) \
      for d in range(norder)]
    for d in range(norder):
      self.xV[d].init_tensor(init_mode = init_mode, rng = rng)

    ## tshape=(self.xviewdimx[d],)
    self.xlambdaU = [tensor_param_cls((self.xviewdimx[d],),norm_axis = 0) \
                       for d in range(norder)] 
    for d in range(norder):
      self.xlambdaU[d].init_tensor(init_mode = 1)

    tshape=(nrank,ndimy)
    self.xQ = tensor_param_cls(tshape,norm_axis=1)
    if ndimy > 1 and self.ilambda_norm == 0:
      self.xQ.init_tensor(init_mode = init_mode, rng = rng)
    else:
      self.xQ.init_tensor(init_mode = 1)

    tshape=(nrank,)
    if self.ilambda_norm == 1:
      self.xlambda = tensor_param_cls(tshape,norm_axis = 0)
    else:
      self.xlambda = tensor_param_cls(tshape,norm_axis = None)      
    self.xlambda.init_tensor(init_mode = 1)


    self.lParams = [self.xU,self.xV,self.xQ,self.xlambda,self.xlambdaU]

    ## init bias
    self.pbias = np.zeros((1,ndimy))
    
    return

  ## ------------------------------------------------
  def init_grad(self):
    """
    Task: to initialize the gradients
    Input:
    Output:
    Modifies: within self.xU, self.xV, self.xQ, self.xlabda, self.xlambdaU 
              self.xTnext # tensor parameter Nesterov accelerated grad offset
                  ## gradient related variables
              self.xTGrad=None     ## tensor gradient
              self.xATGrad=None    ## aggregated(moment) gradient
              self.xnATGrad=None   ## aggregated gradient norm
    """

    for d in range(self.norder):
      self.xU[d].init_tensor_gradient()
      self.xV[d].init_tensor_gradient()
      self.xlambdaU[d].init_tensor_gradient()

    self.xQ.init_tensor_gradient()
    self.xlambda.init_tensor_gradient()

    return

  ## -------------------------------------------------
  def update_parameters(self,**dparams):
    """
    Task: to update the initialized parameters
    Input:  dprams  dictionary { parameter name : value }
    Output:
    Modifies: corresponding parameters
    """

    for key,value in dparams.items():
      if key in self.__dict__:
        self.__dict__[key]=value

    if self.mblock_gap is None:
      self.mblock_gap=self.mblock

    return

  ## ------------------------------------------------
  def normalize_tensorparams(self, ilambda=1, radius=0.75, norm_type=0):
    """
    Task: to project, normalize by L2 norm, the polynomial parameters
                      xP, xQ
                      
    Input: ilambda =1 xlambda[irank], vlambda nglambda is updated
                      with the product of lenght of the parameter
                      vectors before normalization
          radius      ball radius in nrmalization, =1 unit ball
    Output: xnormlambda  the product of lenght of the parameter
                      vectors before normalization
    Modifies:      xP, xQ
                   or
                   xP[irank], xQ[irank], xlambda[irank], vlambda nglambda 
    """

    for d in range(self.norder):
      xnormlambdaU = self.xU[d].normalize(iscale_axis = 0,radius = radius, \
                                            norm_type = norm_type )
      xnormlambda = self.xV[d].normalize(iscale_axis = 0, radius = radius, \
                                           norm_type = norm_type)
      self.xlambdaU[d].normalize(xrescale = xnormlambdaU, inorm_scale = 0, \
                                   radius = radius, norm_type = 0)

    if self.ndimy > 1 and self.ilambda_norm == 0:
      self.xQ.normalize(radius = radius)

    if self.ilambda_norm == 1:
      self.xlambda.normalize(norm_type = 0)

    return

  ## --------------------------------------------
  def update_lambda_matrix_bias(self, lX, Y):
    """
    Task: to compute the initial estimate of xlambda and the bias
    Input:  X     list of 2d arrays, the arrays of input block views
            Y      2d array  of output block
    Output: xlambda  real, the estimation of xlambda
            bias     vector of bias estimation
    """

    norder = self.norder
    m = Y.shape[0]

    xU = self.xU
    xV = self.xV
    xQ = self.xQ.xT
    xlambdaU = self.xlambdaU

    Fr = np.array([None for _ in range(norder)]) 
    for d in range(norder):
      XU = np.dot(lX[d],(xU[d].xT.T*xlambdaU[d].xT).T)
      AXU = self.activation_function(XU,self.lactfunc[d])
      Fr[d] = self.activation_function(np.dot(AXU,xV[d].xT.T), \
                                       self.lactfunc_ext[d])

    if len(Fr) > 1:
      F = np.prod(Fr,0)
    else:
      F = np.copy(Fr[0])

    if self.implicit == 1:   ## implicit function
      ## solve the least square regression problem for xlambda
      ## min_{xlambda} ||np.dot(F,xlambda)||^2, s.t. ||xlambda||_2 = 1
      ## => xlambda is the right singular vector of F 
      ## corresponding to the smallest singular value 
      bias = np.zeros(self.ndimy)
      U,s,V = np.linalg.svd(F,full_matrices = False)
      xlambda  = V[0]
 
    else:
      ## solve the least square regression problem for xlambda
      ## min_{xlambda,bias} ||Y - np.dot(F,np.outer(xlambda,ones(ny)*Q)-bias||^2 
      YQT = np.dot(Y,xQ.T)
      QQT = np.dot(xQ,xQ.T)
      FTF = np.dot(F.T,F)
      f1 = np.sum(F,axis = 0)

      xright = np.sum(F*YQT,axis=0)-np.mean(F*np.sum(YQT,axis=0), axis=0)
      xleft = QQT*(FTF-np.outer(f1,f1)/m)
      xlambda = np.dot(xright, np.linalg.pinv(xleft.astype(float), hermitian = True))
      bias = np.mean(Y - np.dot(F,(xQ.T*xlambda).T), axis=0)
      
    return(xlambda, bias)

  ## --------------------------------------------
  def nag_next(self, gammaoff=0, psign=-1):
    """
    Task: to compute the next point for Nesterov Acclerated gradient
    Input: gamma    =0 no change , gradient scale otherwise
           psign    =0 no change, -1 pull back, +1 push forward
    """

    for d in range(self.norder):
      self.xU[d].nag_offset(gammaoff = gammaoff, psign = psign)
      self.xV[d].nag_offset(gammaoff = gammaoff, psign = psign)
      self.xlambdaU[d].nag_offset(gammaoff = gammaoff, psign = psign)
      
    # if self.ndimy > 1 and self.ilambda_norm == 0:
    self.xQ.nag_offset(gammaoff = gammaoff, psign = psign)
    self.xlambda.nag_offset(gammaoff = gammaoff, psign = psign)
            
    return

  ## -------------------------------------------------------
  def update_parameters_adam(self):
    """
    Task:  to update the parameters of a polynomial, cpoly,
           based on the ADAM additive update
    Input:   
    Modify:  xU,XV,xQ, xAU, xAV,xAQ, xnAU, xnAV,xnAQ, 
             xlambda, xAlambda, xnAlambda,
             xlambdaU, xAlambdaU, xnAlambdaU,
    """

    norder=self.norder

    xnormU=np.zeros(norder)
    xnormV=np.zeros(norder)
    for d in range(norder):
      xnormU[d] = np.linalg.norm(self.xU[d].xTGrad)
      xnormV[d] = np.linalg.norm(self.xV[d].xTGrad)
      
    ## self.store_grad.apprnd(xnorm)
    xmax = np.max(np.vstack((xnormU,xnormV)))
    if xmax > self.max_grad_norm:
      self.max_grad_norm = xmax
      ## print('Grad norm max:',xmax)
    
    if self.sigma*xmax > self.sigmamax:
      self.sigmacorrect = self.sigmamax/(self.sigma*xmax)
      ## print('>>>',self.sigma*xmax,sigmacorrect)
    else:  
      self.sigmacorrect = 1

    for d in range(self.norder):
      self.xU[d].adam_update(opt_params = self)
      self.xV[d].adam_update(opt_params = self)
      self.xlambdaU[d].adam_update(opt_params = self)

    if self.ndimy > 1 and self.ilambda_norm == 0:
      self.xQ.adam_update(opt_params = self)
    if self.ilambda == 1:
      self.xlambda.adam_update(opt_params = self)

    return

  ## ------------------------------------------------
  def quantile_hyperbola(self,f,alpha):
    """
    Task: to compute the hyperbolic quantile approximation
    Input:  f      2d array input 
            alpha  scalar of quantile parameter
    Output: fout   transformed input
    """
    ## derivative of L_1 norm regression via hyperploid approximation
    ## coordinate trasnformation, rotation
    beta_left = np.arctan(1-alpha)
    beta_right = np.arctan(alpha)
    ## yaxis angle after rotation
    beta_yaxis = (np.pi - beta_left - beta_right)/2 + beta_right 
    beta_rotate = beta_yaxis - np.pi/2  ## axis rotation
    ## xaxis angle
    beta_xaxis = beta_right - beta_rotate
    
    ## rotation parameters
    sr = np.sin(beta_rotate) 
    cr = np.cos(beta_rotate)
    ta = np.tan((beta_right+beta_left)/2)
    blin = sr*cr*(1+ta**2)
    ddenom = cr**2 - ta**2 * sr**2
    ## loss:  quantile_scale * ( blin * u + qsqrt)/ddenom 
    ## derivative of the loss: ( blin + ta**2 * u/qsqrt)/ddenom
    ## where: u = f 
    qsqrt = np.sqrt(ta**2*f**2 + ddenom*self.quantile_smooth**2)
    ## derivative
    fout = (blin + ta**2*f/qsqrt)/ddenom

    return(fout)

  ## ------------------------------------------------
  def quantile_logistic(self,f,alpha):
    """
    Task: to compute the logistic quantile approximation
    Input:  f      2d array input 
            alpha  scalar of quantile parameter
    Output: fout   transformed input
    """
    ## logistic function is used to smooth the subgradient of the pinball loss
    ## \frac{1}{1+exp(-t(x-qx0))}-1+alpha
    ## to avoid division by zero, shift to have 0 error at 0
    qx0 = np.log(alpha/(1-alpha))/self.quantile_smooth 
    ## derivative of the pinball loss
    fout = 1/(1 + np.exp(-self.quantile_smooth*(f - qx0))) - (1 - alpha) 

    return(fout)

  ## ------------------------------------------------
  def activation_func(self, f, ifunc=0):
    """
    Task: to compute the value of activation function
    Input:  f      array of input
            ifunc  =0  identity
                   =1  arcsinh  ln(f+(f^2+1)^{1/2})
                   =2  sigmoid  2e^x/(e^x+1)-1
                   =3  tangent hyperbolisc
    Output: F      array of activation values
    """

    if ifunc == 0:
      F = f          ## identity
    elif ifunc == 1: 
      F = gc.log(f + (f**2 + 1)**0.5)   ## arcsinh
    elif ifunc == 2:
      F = 2/(1 + gc.exp(-f)) - 1   ## sigmoid
    elif ifunc == 3: 
      F = gc.tanh(f)  ## tangent hyperbolic
    elif ifunc == 4: ## relu
      F = f*(f>0)
      
    return(F)
  ## ------------------------------------------------
  def activation_func_diff(self, f, ifunc=0, ipar=1):
    """
    Task: to compute the value of the pointwise derivative of
          activation function
    Input:  f      array of input
            ifunc  =0  identity
                   =1  arcsinh  ln(f+(f^2+1)^{1/2})
                   =2  sigmoid  e^x/(e^x+1)
                   =3  tangent hyperbolisc
    Output: DF     array of pointwise drivative of the activation function
    """

    m,n = f.shape
    if ifunc == 0:
      DF = gc.ones((m,n))          ## identity
    elif ifunc == 1: 
      DF = 1/gc.power((gc.power(f,2)+1),0.5)   ## arcsinh
    elif ifunc == 2:               ## sigmoid 
      DF = 2*gc.exp(-f)/(1+gc.exp(-f))**2
    elif ifunc == 3: 
      DF = 1/gc.cosh(f)**2  ## tangent hyperbolic
    elif ifunc == 4:  ## relu
      DF = ipar*(f>0) 
      
    return(DF)

  ## ------------------------------------------------
  def partial_xindex(self, xindex, lsubviews):
    """
    Task: To collect xindex rows where a subset of columns are unique.
          Create index file where the non-unique rows indexed 
          by the corresponding uniques ones.
          FOR IMPICIT FUNCTION LEARNING
    Input: xindex   2d array of join indexes of views
           lsubviews  list of sub-views
    Output: xindex_unique     2d array of rows of xindex with unique
                              indexes corresponding
                              to subset of columns defined by lsubviews
            xindex_reference  vector of indexes connecting non-unique rows 
                              to the unique ones 
    """
    m,n = xindex.shape
    dindex = {}
    xindex_reference = np.zeros(m,dtype = int)
      
    ireference  = 0
    for i in range(m):
      tindex = tuple(xindex[i,lsubviews]) 
      if tindex not in dindex:
        dindex[tindex] = (i,ireference)
        xindex_reference[i] = ireference
        ireference += 1
      else:
        xindex_reference[i] = dindex[tindex][1]
 
    nreference = len(dindex)
    xindex_unique = np.zeros((nreference,n),dtype = int)
    for val in dindex.values():
      xindex_unique[val[1]] = xindex[val[0]]
 
    return(xindex_unique,xindex_reference)

  ## ------------------------------------------------
  def partial_function_value(self, lX, lsubviews):
    """
    Task:  to compute the rank related partial function value
          FOR IMPICIT FUNCTION LEARNING
    Input:  lX        list of 2d array of input data views
            lusbviews   list of views included 
    Output: f    2d array  \circ_{d \in lsubviews} X_d D_{lambdaU} U_d V_d^{T}  
    """

    m = lX[lsubviews[0]].shape[0]

    xU = self.xU
    xV = self.xV
    xlambdaU = self.xlambdaU
    xQ = self.xQ.xT
    xlambda = self.xlambda.xT
    
    F = np.ones((m,self.nrank)) 
    for d in lsubviews:
      if d >= 0 and d < self.norder:
        XU = np.dot(lX[d], (xU[d].xT.T*xlambdaU[d].xT).T)
        AXU = self.activation_function(XU, self.lactfunc[d])
        ## externel activation 
        F *= self.activation_function(np.dot(AXU, xV[d].xT.T), \
          self.lactfunc_ext[d])

    return(F)

  ## ------------------------------------------------
  def function_value(self, lX):
    """
    Task:  to compute the rank related function value
           f=\lambda \circ_r Xp_r q^T +bias
    Input:  lX   list of 2d array of input data views
    Output: f    2d array =\sum  \circ_t XP^(t)T M_{\lambda} Q  +bias  
    """

    m = lX[0].shape[0]

    xU = self.xU
    xV = self.xV
    xlambdaU = self.xlambdaU
    xQ = self.xQ.xT
    xlambda = self.xlambda.xT
    
    F0 = np.ones((m,self.nrank)) 
    for d in range(self.norder):
      XU = np.dot(lX[d], (xU[d].xT.T*xlambdaU[d].xT).T)
      AXU = self.activation_function(XU, self.lactfunc[d])
      ## externel activation 
      F0 *= self.activation_function(np.dot(AXU, xV[d].xT.T), \
        self.lactfunc_ext[d])

    F = np.dot(F0, (xQ.T*xlambda).T)
    F += self.pbias

    return(F)

  ## ------------------------------------------------
  def loss_func(self, Y, F, itransform = 0):
    """
    Task: to compute the block wise loss
    Input:  Y      2d array of true output
            F      2d array of prediction
    Output  loss   real number of loss assumed to be positive
    """

    ## self.lossdegree_internal

    if self.lossdegree_internal > 0 and self.iquantile == 0:
      loss = np.sum(np.abs(Y-F)**self.lossdegree_internal)
    else:
      loss = None

    if loss is not None:
      dloss = 1
      if itransform == 1:
        loss = np.arcsinh(loss/dloss)
      elif itransform == 2:
        loss = np.tanh(loss/dloss)
      
    return(loss)

  ## ------------------------------------------------
  def mb_sampling(self,lxviews,y,ibatch, eps = 0.01):
    """
    Task: to sample the mini-batches based on the loss value
          by Metropolis-Hasting type sampling
          It select between current and previous mini-batch
    Input: lxviews       list of 2d arrays of the views in the mini-batch
           y             2d array of the output realting to the mini-batch
           ibatch        1d array of indexes contained in the mini-batch
           eps           to avoid zero division, has to be >0
    Output:
           imb_select    = 1 if current or = 0 previous is selected    
           lxviewsout    list of 2d arrays the selected inputs
           yout          2d array of selected output
           ibatchout     1d array of selected indexes
    """

    imb_select = 1 ## init selection indicator to curent
    
    if self.imb_sample == 1 and  self.loss_previous is not None:
      f=self.function_value(lxviews)
      loss_current = self.loss_function(y,f)
      if loss_current is not None:
        ## Metropolis-Hasting type update
        p = (loss_current+eps)/(self.loss_previous+eps)
        if p < 1 : 
          rng = np.random.default_rng()
          q = rng.uniform()
          if p <= q:  ## select the previous one
            imb_select = 0
            ## any other cases the current one is selected  

    if imb_select == 1: ## select the current one       
      lxviewsout = [ np.copy(xview) for xview in lxviews]
      yout = np.copy(y)
      ibatchout = np.copy(ibatch)
    else: ## select the previous one
      lxviewsout = [ np.copy(xview) for xview in self.lviews_previous]
      yout = np.copy(self.y_previous)
      ibatchout = np.copy(self.ibatch_previous)
    
    return(imb_select,lxviewsout,yout,ibatchout)
    
  ## ------------------------------------------------
  def gradient(self, lX, y, bias=None, ibatch=None):
    """
    Task:  to compute the gradients for xP, xQ, xlambda
    Input: lX         list of view related 2d arrays of input data block
           y          2d array of output block
           bias       vector (ndimy)
    Output:
    Modifies:  self.xGradU, self.xGradV, self.xGradQ, 
              self.xlambdagrad, self.xlambdagradU
    """
    norder = self.norder
    nrank = self.nrank
    ndimx = self.ndimx
    ndimy = self.ndimy

    m=y.shape[0]
    
    tnsrX = [ self.tensor_load(lX[d]) for d in range(norder)]
    tnsry = self.tensor_load(y)

    xU = [ None for _ in range(norder)]
    xV = [ None for _ in range(norder)]
    xlambdaU = [ None for _ in range(norder)]

    for d in range(norder):
      xU[d] = self.tensor_load(self.xU[d].xTnext)
      xV[d] = self.tensor_load(self.xV[d].xTnext)
      xlambdaU[d] = self.tensor_load(self.xlambdaU[d].xTnext)

    xQ = self.tensor_load(self.xQ.xTnext)
    xlambda = self.tensor_load(self.xlambda.xTnext)
    
    if bias is None:
      bias = self.tensor_load(self.pbias) 
    else:
      bias = self.tensor_load(bias) 

    ## setting the regularization constants
    self.clambda = self.cregular      ## lambda regularization constant

    ## scaling the loss and the regularization
    if self.iscale == 1:
      scale_loss = 1/(m*ndimy)
      scale_lambda = 1/nrank
    else:
      scale_loss = 1
      scale_lambda = 1

    xXUV = [None for _ in range(norder)] 
    xXU = [None for _ in range(norder)] 
    xActD = [None for _ in range(norder)]
    ## external activation
    xActD_ext = [None for _ in range(norder)]
    xActXUV_ext = [None for _ in range(norder)]

    ## Compute the transformations of X by U_d and V_d,
    ## and also by the activation functions 
    for d in range(norder):
      Ulambda = (xU[d].T*xlambdaU[d]).T
      xXU[d] = gc.matmul(tnsrX[d],Ulambda)
      xActD[d] = self.activation_function_diff(xXU[d],self.lactfunc[d])
      xXU[d] = self.activation_function(xXU[d],self.lactfunc[d])
      xXUV[d] = gc.matmul(xXU[d],xV[d].T)
      ## external activation
      xActD_ext[d] = self.activation_function_diff(xXUV[d],self.lactfunc_ext[d])
      xActXUV_ext[d] = self.activation_function(xXUV[d],self.lactfunc_ext[d])

    ## compute the polynomial function 
    F0=xActXUV_ext[0]
    for d in range(1,norder):
       ## \circ_d XM_{lambdaU}U^((d)}V^{(d)}T}M_{\lambda} 
      F0 = F0*xActXUV_ext[d] 

    ## entire predictor function values
    Qlambda = (xQ.T*xlambda).T
    ## error loss handling
    ## the derivative of the loss is needed
    if self.loss_function_diff == None:
       ## negative loss grad, erro
      ferr = (gc.matmul(F0,Qlambda) + bias) - tnsry
    else:
      ## negative loss grad, erro
      ferr = -self.loss_function_diff(tnsry, gc.matmul(F0,Qlambda) + bias, \
        gc, cmodel = self)
    
    if self.implicit == 1:
      ferr = -ferr  ## maximization instead of minimization
    
    ## averaging the loss on the min-batch
    ferr = ferr*scale_loss

    ## data reweighting by error
    if ibatch is not None:
      if self.idweight == 1:
        ## total error
        cferr = np.sum(np.abs(ferr),1)
        ## reweight error
        dw = self.dweight[ibatch]
        dw = m*dw/np.sum(dw)
        ferr = (ferr.T*dw).T 

        ## compute new weights
        xmean = np.mean(cferr)
        dwlin = np.arcsinh(self.dweight_scale*(cferr-xmean))
        dwlin = np.log(dw) + dwlin
        ## bound weights
        iw = np.where(dwlin > self.dweight_max)[0]
        dwlin[iw] = self.dweight_max
        iw = np.where(dwlin < -self.dweight_max)[0]
        dwlin[iw] = -self.dweight_max

        ## update weights
        dw = np.exp(dwlin)
        self.dweight[ibatch] = m*dw/np.sum(dw)

    nE = len(ferr)
    if self.iecum == 1:
      if self.Ecum0 is not None:
        ferr = self.ediscount*ferr + (1-self.ediscount)*self.Ecum0[:nE,:]  
        self.Ecum0[:nE,:] = np.copy(ferr)
      else:
        self.Ecum0 = np.copy(ferr)
    
    if self.iquantile == 0:  ## norm based loss
      if self.lossdegree_internal > 0:
        dferr=gc.linalg.norm(ferr)
        if dferr == 0:
          dferr = 1
        ## degree of the denominator  
        dferr = dferr**(self.lossdegree_internal-2)  
        ferr = ferr*dferr  ## ferr/||ferr||_2^{-(lossdegree_internal-2)}

    elif self.iquantile == 1:  ## smoothed quantile regression 
      alpha = self.quantile_alpha
      ## we need only the direction of the gradient
      # dferr=gc.linalg.norm(ferr)
      # if dferr == 0:
      #   dferr = 1
      # ferr = ferr/dferr  ## ferr/||ferr||
      
      if self.iquantile_hyperbola == 1:
        ferr = self.quantile_hyperbola(ferr,alpha)
      else:  
        if alpha <= 0:
          alpha = 0.0001
        if alpha >= 1:
          alpha = 0.999
        ferr = self.quantile_logistic(ferr,alpha)
 
    ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    ## computation the gradients

    ## computing more than ones occuring terms and factors to save time
    ferrQT = gc.matmul(ferr,xQ.T)

    ## gradient not depending on degree
    self.xQ.xTGrad = self.tensor2np((gc.matmul(F0.T,ferr).T*xlambda).T)
    self.xlambda.xTGrad = self.tensor2np(gc.sum(F0*ferrQT,0))
        
    ## compute F_{\subsetminus d}
    if norder > 1:
      for d in range(norder):
        ipx = np.arange(norder-1)
        if d < norder-1:
          ipx[d:] += 1
        Zd=xActXUV_ext[ipx[0]]
        for di in range(1,len(ipx)):
          Zd = Zd*xActXUV_ext[ipx[di]]    ## Z^{(d)}

        dEQF = Zd*ferrQT*xActD_ext[d]      ## external activation included
        dEQFVH = xActD[d]*gc.matmul(dEQF,(xV[d].T*xlambda).T)
        dXEQFVH = gc.matmul(tnsrX[d].T,dEQFVH)

        self.xV[d].xTGrad = self.tensor2np((gc.matmul(dEQF.T,xXU[d]).T*xlambda).T)
        self.xU[d].xTGrad = self.tensor2np((dXEQFVH.T*xlambdaU[d]).T)
        self.xlambdaU[d].xTGrad = self.tensor2np(gc.sum(dXEQFVH*xU[d],1))

    else:
      ## Zd is empty, or = ones(m,nrank)
      d = 0
      dEQF = ferrQT*xActD_ext[d]   ## external activation included

      dEQFVH = xActD[d]*gc.matmul(dEQF,(xV[d].T*xlambda).T)
      dXEQFVH = gc.matmul(tnsrX[d].T,dEQFVH)

      self.xV[d].xTGrad = self.tensor2np((gc.matmul(dEQF.T,xXU[d]).T*xlambda).T)
      self.xU[d].xTGrad = self.tensor2np((dXEQFVH.T*xlambdaU[d]).T)
      self.xlambdaU[d].xTGrad = self.tensor2np(gc.sum(dXEQFVH*xU[d],1))

    ## regularization terms for xlambda
    if self.ilambda == 1:
      cxlambda = self.tensor2np(xlambda)
      self.xlambda.xTGrad -= self.clambda*scale_lambda \
        *(np.sign(cxlambda)*np.abs(cxlambda)**(self.regdegree - 1))
          
    return

  ## ------------------------------------------------
  def incremental_step(self, lX, y, icount, ibatch=None):
    """
    Task: to compute one step of the iteration
    Input: lX       list of view related  2d arrays of input block
           y        2d array of output block
           icount     block index
    Modifies: via the called functions, xP,xQ and the gradient related variables
    """
    
    if self.inormalize == 1:
      self.normalize_tensorparams(radius = self.radius, \
                                  norm_type = self.norm_type)

    ## if first parameter =0 then no Nesterov push forward step
    self.nag_next(gammaoff = self.gammanag, psign=-1)
    
    self.gradient(lX,y,bias = self.pbias, ibatch = ibatch)

    ## self.update_parameters_nag()
    self.update_parameters_adam()

    f=self.function_value(lX)

    ## bias is averaged on all processed block and data
    if self.ibias == 1:
      prev_bias = self.pbias
      self.pbias = np.mean(y - f,0)/(icount + 1)+prev_bias*icount/(icount + 1)
      f += (self.pbias - prev_bias)  
        
    return(f)

  ## --------------------------------------------------
  def fit(self, lXtrain, Ytrain, llinks=None, xindex=None, yindex=None, \
            nepoch=10, idataadd=0, imultitask=0, lrankuv = None):
    """
    Task: to compute the full training cycle
    Input: lXtrain       list of 2d arrays of training input data 
                         corresponding to the views, 
                         one view might have more than one data array
           Ytrain        2d array of training output
           llink         if not None, then list of lists, 
                         the outer list enumerate the views
                         in the inner list the data arrays corresponding 
                         to one view are listed. 
           xindex        if is not None then 2d array of indexes to join
                         the views in the mini-batches, 
                         The order of the columns in xindex has to be the same 
                         as in llinks. 
            yindex       if is not None then it is a vector of indexes of
                         output relative to Ytrain. yindex has to have the
                         length mtrain.
                         if yindex is given then Ytrain could be an arbitrary 
                         array in length
           nepoch        number of repetation of the iteration steps
           idataadd      =1 new data is added to the existing model
           imultitask    =0 parameters and gradients are initialized
                         =1 parameters and gradients inherited from
                         the previous runs
          lrankuv        links of rankuv values for each view
                         if it is None then
                             lself.nrankub = [rankuv for _ range(nview)]]
                         if len(lrankuv)< nview then it is padded with nrankuv
                         if len(lrankuv)> nview
                             then only first nview elements are used
    Output:  self        reference to the self object
    Modifies:  computes the polynomial parameters xP,xQ, xlambda,xbias
    """

    self.lossdegree_internal = self.lossdegree + 2 
    
    if idataadd == 0:
      self.ifirstrun=1
      self.nrank=self.nrank0     ## initialize rank for rank extensions
      self.xviewdimx = None

    if nepoch is not None:  
      self.nrepeat=nepoch

    ## to be back compatible
    if not isinstance(lXtrain,list): ## lXtrain is not list
      if isinstance(lXtrain,np.ndarray):
        lXtrain = [lXtrain]   ## makinkg list of arrays
        if llinks is None:
          llinks = [ 0 for _ in range(self.norder)]

    ## check the input matrices, are they 2d arrays
    for i in range(len(lXtrain)):
      if len(lXtrain[i].shape) == 1:
        lXtrain[i]=lXtrain[i].reshape((lXtrain[i].shape[0],1))

    self.ninput = len(lXtrain)

    ## set default links where all input matrices correspond to one view
    if llinks is None:
      llinks = [ [i] for i in range(self.ninput)]
    
    if xindex is None:  ## online mini-batches
      ## all shape[0] need to be the same
      xshape0 = np.array([ lXtrain[i].shape[0] for i in range(self.ninput)])
      if np.sum(xshape0 != xshape0[0]) != 0:  ## shape error
        print('ERROR:' +  \
          'If xindex is None then all arrays in lXtrain have to have the same lenght!') 
        return
      
    if xindex is not None:
      if xindex.shape[1] < len(llinks):
        print('ERROR:'+ \
          'Number of columns in xindex has to be greater or equal to the length of llinks!')
        return

    if xindex is None:
      mtrain = lXtrain[0].shape[0]
    else:
      mtrain = len(xindex)

    self.dweight = np.ones(mtrain)  ## data weighting  

    if yindex is not None:
      if len(yindex) != mtrain:
        print('ERROR:'+ \
          'Length of yindex has to be equal to the lenght of the full training!')
        return
        
    ## reshape output vector into matrix
    if len(Ytrain.shape) > 1:
      self.ndimy = Ytrain.shape[1]
    else:
      Ytrain = Ytrain.reshape((mtrain,1))
      self.ndimy = 1

    ## output scaling and normalization
    self.compute_output_scaling(Ytrain)

    self.xinputdimx = np.array([ lXtrain[i].shape[1] for i in range(self.ninput)])
      
    ## compute input means and scaling
    self.compute_input_scaling(lXtrain)

    ## construct the views out of the input matrices
    self.norder = len(llinks)
    self.nview = len(llinks) 
    ## complete the llinks 
    self.llinks = [ self.links_item_completion(litem, \
      self.ninput) for litem in llinks]
    
    ## collect the number of columns of the views in self.xviewdimx
    self.process_view_list(lXtrain, self.llinks, iconcat = 0)

    ## set the lrankuv
    if lrankuv is None:
      self.lrankuv = [ self.nrankuv for _ in range(self.nview)]
    else:
      self.lrankuv = lrankuv
    if len(self.lrankuv) < self.nview:
      self.lrankuv = self.lrankuv + [ self.nrankuv for _ in \
        range(self.nview - len(self.lrankuv))]
    if len(self.lrankuv) > self.nview:
      self.lrankuv = self.lrankuv[:self.nview]
    
    ## initialize activation functions
    if self.lactfunc is None:
      self.lactfunc = [self.iactfunc for _ in range(self.nview)] 
    if self.lactfunc_ext is None:
      self.lactfunc_ext = [self.iactfunc_ext for _ in range(self.nview)] 

    ## parameters initialized only in the first run
    ## additional data or rank run on the herited parameters
    if idataadd  ==  0 and imultitask == 0:  
      self.init_tensorparams()
      self.init_grad()
      self.max_grad_norm=0.0
      
    icount = 0
    ifirst = 1   ## for a new rank iteration initialization is needed        
    self.iter = 1  ## iteration counter
    xselect = np.arange(mtrain)  ## mini-bach selection indexes
    self.sigma = self.sigma0   ## initial learning rate
    self.ediscount = self.ediscount0 ## set the function discount over the mini-batches

    for irepeat in range(self.nrepeat):

      self.sigma -= self.sigma**2/self.dscale  ## learning rate update
      nblock=0
      self.iter = 1  ## iteration counter reinitialization
      ## blocks might be randomly chosen 
      if self.irandom == 1:
        if self.rng is None:
          rng = np.random.default_rng()
        else:
          rng = self. rng
        rng.shuffle(xselect)
      
      ## compute block lenght for chunk blocks
      if self.mblock_gap is None:
        mblock_gap = self.mblock
      else:
        mblock_gap = self.mblock_gap

      for iblock in range(0,mtrain,mblock_gap):

        self.icount = icount
        
        if iblock+self.mblock<=mtrain:
          mb = self.mblock
        else:
          mb = mtrain - iblock
        if mb == 0:
          break

        ## load random block
        ib = np.arange(iblock,iblock + mb)
        iib = xselect[ib]   ## indexes of the minibatch

        ## load the mini-batches
        lx_b = self.preprocess_minibatch(lXtrain, self.llinks, iib, xindex)

        ## the output block is simple
        
        if yindex is None:
          y_b = Ytrain[iib]
        else:
          y_b = Ytrain[yindex[iib]]
        if sp.sparse.isspmatrix_csr(y_b):
          y_b = y_b.toarray()

        ly_b = [y_b]
        ## list is used to avoid copying the matrix
        self.output_scaling(ly_b) ## scaling the output
        y_b = ly_b[0]

        ## in the first iteration estimate lambda and the bias
        if ifirst == 1 and imultitask == 0:
          self.normalize_tensorparams(radius = self.radius, \
                                      norm_type = self.norm_type )
          xlambda,bias = self.update_lambda_matrix_bias(lx_b,y_b)
          if self.ilambda == 1:
            self.xlambda.xT = xlambda
          prevlambda = xlambda
          self.pbias = bias
          ifirst = 0

        if icount >= self.mb_sample_delay:   
          imb_select, lx_b, y_b, iib = self.mb_sampling(lx_b,y_b,ibatch = iib)
        f = self.incremental_step(lx_b,y_b,icount,ibatch = iib)
        ## store the views and the loss
        self.lviews_previous = [ np.copy(xview) for xview in lx_b]
        self.y_previous = np.copy(y_b)
        self.ibatch_previous = np.copy(iib)
        self.loss_previous = self.loss_function(y_b,f) 

        ## print iteration states
        if icount%self.report_freq == 0:
          deye=np.linalg.norm(self.xlambda.xnATGrad)
          if np.std(y_b) == 0 or np.std(f) == 0:
            fcorr = 0
          else:
            fcorr = np.corrcoef(f.ravel(),y_b.ravel())[0,1]

          print(self.nrank,icount,irepeat,iblock, \
                '%7.4f'%np.linalg.norm(f-y_b), \
                '%7.4f'%fcorr, \
                '%8.2f'%(np.linalg.norm(self.xlambda.xT)),
                '%8.2f'%(np.linalg.norm(self.xQ.xT)),
                end='')
          xUnorm = 0
          xVnorm = 0
          xlambdaUnorm = 0
          for d in range(self.norder):
            xUnorm+=np.sqrt(np.sum(self.xU[d].xT**2))
            xVnorm+=np.sqrt(np.sum(self.xV[d].xT**2))
            xlambdaUnorm+=np.sqrt(np.sum(self.xlambdaU[d].xT**2))

          print('%8.2f'%xUnorm, \
                '%8.2f'%xVnorm, \
                '%8.2f'%xlambdaUnorm, \
                '%8.2f'%deye)  
          sys.stdout.flush()

        ## reduce the decay of the learning speed,
        ## diminish the set size only after self.nsigma iterations 
        if icount%self.nsigma == 0:
          self.sigma -= self.sigma**2/self.dscale
        icount += 1
        nblock += 1

      self.iter += 1
      prevlambda=self.xlambda
      sys.stdout.flush()
          
    print('icount:',icount)
    self.ifirstrun = 0   ## first run is finished

    self.fitfinished = 1  ## mark that fit has been executed
    
    return(self)

  ## ---------------------------------------------
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ## If Xtest, Ytrain are sparse they need to converted into dense 2darray
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  def predict(self, lXtest, Ytrain=None, llinks=None, xindex=None, itestmode=0):
    """
    Task: to compute the predictions
    Input: lXtest      list of 2d arrays of test input data 
                       corresponding to the views, 
                       one view might have more than one data array
           Ytrain      2d array of training outputs, it is not None then the
                       prediction is chosen as the best training output vector
           llink       if not None, then list of lists, 
                       the outer list enumerate the views
                       in the inner list the data arrays corresponding 
                       to one view are listed. 
           xindex      if not None then 2d array of indexes to join the
                       views in the mini-batches, 
                       The order of the columns in xindex has to be the same 
                       as in llinks. 
           itestmode   test mode =0 take the best training output
                       =1  set the largest predicted output components
                       to 1 and the others to 0. 
    """

    if self.llinks is None:
      print('fit() has to be applied before predcition!')
      return
    else:
      llinks = self.llinks

    ## to be back compatible
    if not isinstance(lXtest,list): ## lXtest is not list
      if isinstance(lXtest,np.ndarray):
        lXtest = [lXtest]   ## makinkg list of arrays
        if llinks is None:
          llinks = [ 0 for _ in range(self.norder)]
    
    ## check the input matrices, are they 2d arrays
    for i in range(len(lXtest)):
      if len(lXtest[i].shape) == 1:
        lXtest[i] = lXtest[i].reshape((lXtest[i].shape[0],1))

    ## process input matrices
    ninput = len(lXtest)
    if ninput != self.ninput:
      print('The number of input matrices in the test has to be the' + \
                   'same as in the training!')
      return

    ## set default links where all input matrices correspond one view
    if llinks is None:
      llinks = [ [i] for i in range(ninput)]
    
    if len(llinks) != self.nview:
      print('The number of input views in the test has to be the' + \
                   'same as in the training!')
      return

    if xindex is not None:
      if xindex.shape[1] < len(llinks):
        print('ERROR:'+ \
          'Number of columns in xindex has to be greater or equal to the length of llinks!')
        return

    if xindex is None:
      mtest = lXtest[0].shape[0]
    else:
      mtest = len(xindex)

    ## construct the views out of the input matrices
    self.norder = len(llinks)
    if xindex is None:  ## online mini-batches
      ## all shape[0] need to be the same
      xshape0 = np.array([ lXtest[i].shape[0] for i in range(self.ninput)])
      if np.sum(xshape0 != xshape0[0]) != 0:  
        ## shape error
        print('ERROR:' +  \
          'If xindex is None then all arrays in lXtrain have to have same lenght!') 
        return

    if Ytrain is not None: 
      if len(Ytrain.shape) == 1:
        Ytrain = Ytrain.reshape((len(Ytrain),1))
      ynorm1 = np.sqrt(np.sum(Ytrain**2,1))
      ynorm1 += 1*(ynorm1 == 0)

    ## the prediction dimension
    Ypred = np.zeros((mtest,self.ndimy))

    mblock = self.ntestblock * self.mblock
    for iblock in range(0,mtest,mblock):

      if iblock + mblock <= mtest:
        mb = mblock
      else:
        mb = mtest - iblock
      if mb == 0:
        break
      irange = np.arange(iblock,iblock+mb)  ## indexes in the minibatch

      ## load the mini-batches
      lxblocks = self.preprocess_minibatch(lXtest, self.llinks, irange, xindex)

      ## list to avoid copying the array
      lyblock = [self.function_value(lxblocks)]
      self.output_rescaling(lyblock)
      yblock = lyblock[0]

      if Ytrain is None:  ## direct prediction, regression
        Ypred[irange] = yblock
      else:
        ## prediction based on the most similar training example
        if itestmode == 0:
          if yblock.shape[1] > 1:
            ynorm2 = np.sqrt(np.sum(yblock**2,1))
            zscore = np.dot(Ytrain,yblock.T)
            zscore /= np.outer(ynorm1,ynorm2)
            iscore = np.argmax(zscore,0)    ## blocks might be randomly chosen 
            Ypred[irange] = Ytrain[iscore]

        elif itestmode == 1:
          if yblock.shape[1] > 1:
            xmax = np.max(yblock,1)
            Ypred[irange] = 2*(yblock >= np.outer(xmax,np.ones(self.ndimy)))-1
          else:  ## ndimy == 1
            Ypred[irange] = np.sign(yblock)
      
    return(Ypred)

  ## --------------------------------------------------
  def export_latent_views(self, lXtrain, xindex=None, ifullview = 0):
    """
    Task: to compute the full training cycle
    Input: lXtrain       list of 2d arrays of training input data 
                         corresponding to the views, 
                         one view might have more than one data array
           xindex        if is not None then 2d array of indexes to join
                         the views in the mini-batches, 
                         The order of the columns in xindex has to be the same 
                         as in llinks.
           ifulview      =0 latent views extracted only for the unique indexes of xindex,
                         =1 latent view is extracted to all repeated index of the xindex 
    Output: lviews       list of 2d arrays of the latent views,
                         order is the same as in the llinks               
    """

    if self.llinks is None:
      print('export_latent_views() has to be applied after fit!')
      return
    else:
      llinks = self.llinks

    ## to be back compatible
    if not isinstance(lXtrain,list): ## lXtrain is not list
      if isinstance(lXtrain,np.ndarray):
        lXtrain = [lXtrain]   ## makinkg list of arrays

    ## check the input matrices, are they 2d arrays
    for i in range(len(lXtrain)):
      if len(lXtrain[i].shape) == 1:
        lXtrain[i]=lXtrain[i].reshape((lXtrain[i].shape[0],1))

    self.ninput = len(lXtrain)
    
    if xindex is None:  ## online mini-batches
      ## all shape[0] need to be the same
      xshape0 = np.array([ lXtrain[i].shape[0] for i in range(self.ninput)])
      if np.sum(xshape0 != xshape0[0]) != 0:  ## shape error
        print('ERROR:' +  \
          'If xindex is None then all arrays in lXtrain have to have the same lenght!') 
        return
      
    if xindex is not None:
      if xindex.shape[1] < len(llinks):
        print('ERROR:'+ \
          'Number of columns in xindex has to be greater or equal to the length of llinks!')
        return

    if xindex is None:
      mtrain = lXtrain[0].shape[0]
    else:
      mtrain = len(xindex)
                
    ## enumerate the views
    lviews = [ None for _ in range(self.nview)]
    for iview in range(self.nview):

      ## extract the range of a view
      m = len(xindex)

      if ifullview == 0:
        nmax = np.max(xindex[:,iview])
        drange = {}
        for i  in range(m):
          irow = xindex[i,iview]
          if irow not in drange:
            drange[irow] = i
          if len(drange) > nmax:   ## stop if every index up to the maximum has been found
            break

        iviewrange = np.array([ drange[key] for key in drange.keys()])
        iviewrange = np.sort(iviewrange)
        mrange = len(iviewrange)

      else: ## full views
        iviewrange = xindex[:,iview]
        mrange = len(xindex) 

      xview = np.zeros((mrange,self.nrank))
      ## compute block wise to reduce the size of the arrays in the operations, -> GPU
      mblock = self.ntestblock * self.mblock
      for iblock in range(0,mrange,mblock):

        if iblock + mblock <= mrange:
          mb = mblock
        else:
          mb = mrange - iblock
        if mb == 0:
          break
        irange = np.arange(iblock,iblock+mb)  ## indexes in the minibatch
      
        ## load the mini-batches
        lsubviews = [iview]
        lX_view = self.preprocess_minibatch(lXtrain, self.llinks, irange, \
                  xindex, lsubset = lsubviews)
        xview[irange] = self.partial_function_value(lX_view, lsubviews)

      lviews[iview] = xview
      
      print('View processed:',iview)

    return(lviews)

  ## ---------------------------------------------------
  def save_parameters(self, filename):
    """
    Task: to save the parameters learned
    Input:   filename   filename of the file used to store the parameters
    """

    savedict={}
    for var in self.lsave:
      value=self.__dict__[var]
      savedict[var]=value

    fout=open(filename,'wb')
    pickle.dump(savedict,fout)
    
    return

  ## ---------------------------------------------------
  def load_parameters(self, filename):
    """
    Task: to load the parameters learned and saved earlier
    Input:    filename   filename of the file used to store the parameters
    Modifies: (self) norder, nrank,iyscale,yscale,ldim,ndimy
                     xP,xQ,xlambda,pbias
    """

    fin=open(filename,'rb')
    savedict=pickle.load(fin)

    for var in self.lsave:
      if var in self.__dict__:
        self.__dict__[var]=savedict[var]
      else:
        print('Missing object attribute:',var)

    return

  ## ----------------------------------------------------------
## ###################################################
## ################################################################
## if __name__  ==  "__main__":
##   if len(sys.argv) == 1:
##     iworkmode=0
##   elif len(sys.argv)>=2:
##     iworkmode=eval(sys.argv[1])
##   main(iworkmode)
