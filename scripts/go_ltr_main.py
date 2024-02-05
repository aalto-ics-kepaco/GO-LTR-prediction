## LTR example main file
import sys
import time

import numpy as np
## ###################################################
## for demonstration
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmplot
import sklearn.model_selection 
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from itertools import chain, combinations
## ###################################################

import ltr_solver_multiview_0164_dev as ltr

## ################################################################
## ################################################################
class ltr_cls:


	## --------------------------------
	def __init__(self, **parameters):

		self.llinks = None
		self.cmodel = ltr.ltr_solver_cls(norder = parameters['order'], \
																		 rank = parameters['rank'], \
																		 rankuv = parameters['rankuv'], \
																		 loss_function_diff = parameters['loss_diff'])

		## self.cmodel.nrepeat = parameters['nepoch']
		self.__set_basic_parameters()
		self.__set_additional_parameters()
		
		return

	## public methods  
	## --------------------------------
	def set_params(self,**hyperparams):

		if 'order'  in hyperparams:
			norder = hyperparams['order']
		else:
			norder = self.cmodel.norder
			
		if 'rank' in hyperparams:
			nrank = hyperparams['rank']
		else:
			nrank = self.cmodel.nrank0
		
		self.cmodel.update_parameters(norder = norder, \
										nrank0 = nrank, \
										nrank = nrank, \
										nrankuv = nrank)

		return

	## --------------------------------
	def fit(self,lX,Y, llinks = None, xindex = None, nepoch = None):

		self.llinks = llinks
		if nepoch is None:		
			epochset = self.cmodel.nrepeat
		else:
			epochset = nepoch
			
		self.cmodel.fit(lX,Y, llinks = self.llinks, xindex = xindex, \
										nepoch = epochset)

		return(self)

	## --------------------------------
	def predict(self,lXtest, Ytrain = None, llinks = None, xindex = None, itestmode = 0):

		Ypred = self.cmodel.predict(lXtest, Ytrain = Ytrain, llinks = llinks, \
			xindex = xindex, itestmode = itestmode)
		
		return(Ypred)

	## --------------------------------
	def export_latent_views(self,lX, llinks = None, xindex = None):

		lviews = self.cmodel.export_latent_views(lX, xindex = xindex)

		return(lviews)
		
	## ###########################################################  
	## private methods
	## ----------------------------------------
	def __set_basic_parameters(self):

		cmodel = self.cmodel

		## -------------------------------------
		## Parameters to learn
		## the most important parameter
		norder=1      ## maximum power, valid if no design llinks = None
		rank=1000       ## number of ranks
		rankuv=600      ## internal rank for bottlenesck if rankuv<rank
		sigma=0.01    ## learning step size
		nsigma=1      ## step size correction interval
		gammanag=0.9     ## discount for the ADAM method
		gammanag2=0.9    ## discount for the ADAM method norm

		# mini-batch size,
		mblock=500

		## number of epochs
		nrepeat=10

		## regularization constant for xlambda optimization parameter
		cregular=1 #0.000001 #1

		## activation function
		iactfunc = 4  ## =0 identity, =1 arcsinh, =2 2*sigmoid-1, =3 tanh, =4 relu
		iactfunc_ext = 0  ## =0 identity, =1 arcsinh, =2 2*sigmoid-1, =3 tanh, =4 relu

		## cmodel.lossdegree = 0  ## =0 L_2^2, =1 L^2, =0.5 L_2^{0.5}, ...L_2^{z}
		lossdegree = 0  ## default L_2^2 =0
		regdegree = 1   ## regularization degree, Lasso

		norm_type  = 0  ## parameter normalization
										## =0 L2 =1 L_{infty} =2 arcsinh + L2  
										## =3 RELU, =4 tanh + L_2 

		perturb = 0.0   ## gradient perturbation

		report_freq = 100  ## frequency of the training reports

		## set optimization parameters
		cmodel.update_parameters(mblock = mblock, \
										sigma0 = sigma, \
										nsigma = nsigma, \
										gammanag = gammanag, \
										gammanag2 = gammanag2, \
										## nrepeat = nrepeat, \
										cregular = cregular, \
										iactfunc = iactfunc, \
										iactfunc_ext = iactfunc_ext, \
										lossdegree = lossdegree, \
										regdegree = regdegree, \
										norm_type  = norm_type, \
										perturb  =  perturb, \
										report_freq  =  report_freq)
		
		return
		
	## ----------------------------------------  
	def __set_additional_parameters(self):
		"""
		Taks to set additional LTR parameters,
				 in the basic case they might not be changed at all.
				 In this version the default values are used. 
		Input: cmodel  reference of the solver object
		"""

		cmodel = self.cmodel
		
		## normalization
		## output
		cmodel.iymean=0      ## =1 output vectors centralized
		cmodel.ymean=0        ## ymean
		cmodel.iyscale=1      ## =1 output vectors are scaled
		cmodel.yscale=0       ## the scaling value
		## input
		cmodel.ixmean = 1   ## centralized inputs by mean
		cmodel.ixscale = 1  ## scale inputs by L_infty norm 
		cmodel.ixl2norm = 0 ## scale inputs by l_2 norm
		cmodel.ihomogeneous=1  ## =1 input views are homogenised =0 not 

		## quantile regression
		cmodel.iquantile = 0   ## norm based regression, =1 quantile regression
		cmodel.quantile_alpha = 0.5  ## quantile, confidence, parameter
		cmodel.iquantile_hyperbola = 1  ## = hyperbola = 0 logistic approximation
		cmodel.quantile_smooth = 0.2    ## smoothing parameter of the pinball loss
									 ## in case of logistic 
									 ## if it is larger then it is closer to the pinball 
									 ## but less smooth 
									 ## in case of hyperbola
									 ## if it is smaller then it is closer to the pinball 
									 ## but less smooth
		
		cmodel.quantile_scale =1  ## scale the tangent direction 

		return

## #####################################################    
def report_params(cmodel):

	## report the most important parameter values
	print('Order:',cmodel.norder)
	print('Rank:',cmodel.nrank)
	print('Rankuv:',cmodel.nrankuv)
	print('Step size:',cmodel.sigma0)
	print('Step freq:',cmodel.nsigma)
	print('Step scale:',cmodel.dscale)
	print('Epoch:',cmodel.nrepeat)
	print('Mini-batch size:',cmodel.mblock)
	print('Discount:',cmodel.gamma)
	print('Discount for NAG:',cmodel.gammanag)
	print('Discount for NAG norm:',cmodel.gammanag2)
	print('Bag size:',cmodel.mblock)
	print('Regularization:',cmodel.cregular)
	print('Gradient max ratio:',cmodel.sigmamax)
	print('Type of activation:',cmodel.iactfunc)
	print('Degree of loss:',cmodel.lossdegree)
	print('Degree of regularization:',cmodel.regdegree)
	print('Normalization type:',cmodel.norm_type)
	print('Gradient perturbation:', cmodel.perturb)
	print('Activation:', cmodel.iactfunc)
	print('Input centralization:', cmodel.ixmean)
	print('Input L_infty scaling:', cmodel.ixscale)
	print('Quantile regression:',cmodel.iquantile)
	print('Quantile alpha:',cmodel.quantile_alpha)    ## 0.5 for L_1 norm loss
	print('Quantile smoothing:',cmodel.quantile_smooth)

	return
		
## ################################################################
class loss_diff_cls:
	"""
	Task:  to define the external loss function derivative
	"""

	def __init__(self, stype='least_square'):
		self.stype=stype
		print("Loss function: ", self.stype)

		## for test
		return

	## -------------------------------------------------
	def loss_diff(self,Y,F,gc, cmodel = None):
		"""
		Task to compute the derivative of the loss
		Input:  Y         array(1d, 2d) of real response
						F         array(1d, 2d) of predicted response
						gc        reference to numerical package, e.g. numpy or pytorch
						cmodel    ltr object reference
		Output  xlossdev  gradient(Jacobian) array
		"""

		ltypes = ['least_square', 'L1', 'tanh_smooth_L1','arcsinh_log', \
			'huber_square', 'seal'] #SEAL loss - Structured energy network as loss global energy
		##print("cmodel.icount: ", cmodel.icount)

		stype = self.stype#'' #huber_square', default 'least_square', arcsinh_log
		if stype == 'least_square':
			xlossdev = Y - F   ## 1/2 ||Y-F||_2^2
		elif stype == 'L1':
			xlossdev = gc.sign(Y - F)   ## ||Y-F||_1
		elif stype == 'tanh_smooth_l1':
			xlossdev = gc.tanh(Y - F)   ## approximate L1 norm, ~||Y-F||_1
		elif stype == 'arcsinh_log':
			scale = 5#10
			xlossdev = gc.arcsinh(scale*(Y - F))   ## approximate log
		elif stype == 'huber_square':
			margin_scale = 0.7#0.5
			ymin = np.min(Y)
			ymax = np.max(Y)
			beta = np.mean(Y)
			gammamax = margin_scale*(ymax - beta)
			gammamin = margin_scale*(beta -ymin)
			
			m,n = Y.shape     
			xlossdev = np.zeros((m,n))
			iy  = np.where((Y < beta - gammamin) * (F > beta - gammamin) == True )
			xlossdev[iy] = beta - gammamin - F[iy]
			iy  = np.where((Y > beta + gammamax) * (F < beta + gammamax) == True )
			xlossdev[iy] = beta + gammamax - F[iy]
		
		
		elif stype == 'huber_lin':  ## smoothed Huber
			delta = 20  ## steepness of the derivative at 0
			
			margin_scale = 0.7
			ymin = np.min(Y)
			ymax = np.max(Y)
			beta = np.mean(Y)
			gammamax = margin_scale*(ymax - beta)
			gammamin = margin_scale*(beta -ymin)
			m,n = Y.shape

			xlossdev = np.zeros((m,n))
			iy  = np.where((Y < beta - gammamin) )
			xlossdev[iy] = - (np.tanh(delta*(F[iy] - beta + gammamin))+1)/2
			iy  = np.where((Y > beta + gammamax) )
			xlossdev[iy] = (np.tanh(-delta*(F[iy] - beta -  gammamin))+1)/2
			
		elif stype == 'huber_power':
			huberpower = 1.1#2   ## degree of the loss = 1 + huberpower
			
			margin_scale = 0.7
			ymin = np.min(Y)
			ymax = np.max(Y)
			beta = np.mean(Y)
			gammamax = margin_scale*(ymax - beta)
			gammamin = margin_scale*(beta -ymin)
			m,n = Y.shape
			
			eps = 0#1e-6

			xlossdev = np.zeros((m,n))
			iy  = np.where((Y < beta - gammamin) * (F > beta - gammamin) == True )
			xlossdev[iy] = -(F[iy] - beta + gammamin + eps)**huberpower
			iy  = np.where((Y > beta + gammamax) * (F < beta + gammamax) == True )
			xlossdev[iy] = (-(F[iy] - beta - gammamax) + eps)**huberpower
			
		elif stype == 'seal':
			epsilon = 1e-3
			xlossdev = gc.arcsinh(5*(Y - F)) + (gc.exp(F)/(1+gc.exp(F)))

		return(xlossdev)

## ***************************************
def centr_kern(Ktrain, Ktest=None, train=True):
	"""
	Centers the kernel matrix
	"""
	(m,n)=Ktrain.shape
	if not train:	
		(mt,n)=Ktest.shape
	
	if train:
		K = Ktrain - np.outer(np.ones(m),np.mean(Ktrain,axis=0)) \
			 - np.outer(np.mean(Ktrain,axis=1),np.ones(n)) \
			 + np.ones((m,n))*np.mean(Ktrain)
			
	else:
		K = Ktest - np.outer(np.ones(mt),np.mean(Ktrain,axis=0)) \
			 - np.outer(np.mean(Ktest,axis=1),np.ones(n)) \
			 + np.ones((mt,n))*np.mean(Ktrain)
					
	return K    
## ################################################################

## ################################################################
def acc_eval(yobserved,ypredicted):
	"""
	Task: to report some statistics
	Input: yobserved    2d array of {-1,+1,0} data
				 ypredicted    2d array of {-1,+1,0} prediction
	Output: prec         precision
					recall       recall
					f1           f1 measure
					acc          tag based accuracy
					xaccur       confusion matrix between {-1,0,+1}
	"""

	## nobject,nclass=yobserved.shape

	tp=np.sum((yobserved>0)*(ypredicted>0))
	fp=np.sum((yobserved<=0)*(ypredicted>0))
	fn=np.sum((yobserved>0)*(ypredicted<=0))
	tn=np.sum((yobserved<=0)*(ypredicted<=0))

	print('Tp,Fn,Fp,Tn:',tp,fn,fp,tn)

	if tp+fp>0:
		prec=tp/(tp+fp)
	else:
		prec=0
		
	if tp+fn>0:
		recall=tp/(tp+fn)
	else:
		recall=0
	
	if prec+recall>0:
		f1=2*prec*recall/(prec+recall)
	else:
		f1=0

	# Compute ROC curve and ROC area for each class
	#fpr, tpr, _ = roc_curve(yobserved.flatten(), ypredicted.flatten())
	#roc_auc = np.round(auc(fpr, tpr),4)
	print("Precision: ", np.round(prec,4))
	print("Recall: ", np.round(recall,4))
	print("F1 score: ", np.round(f1,4))
	
	return
## ################################################################
## ################################################################
## test
def generate_multiview_configurations(feature_list):
	"""
	returns a list of feature combinations for the various multiview feature 
	combinations (singleview, two-view, three-view)
	"""
	combs = ((combinations(feature_list, r)) for r in range(1, len(feature_list) + 1))
	list_combinations = list(chain.from_iterable(combs))
	multiviews = [list(list_combinations[i]) for i in range(len(list_combinations))]
	return multiviews

def create_feature_list(ont_sdir, fold_idx, ppi_mode):
	"""
	creates the paths for the respective data sources and returns a list of them 
	"""
	if ppi_mode == "transductive":
		train_interpro_mat = ont_sdir+"/data_splits/interpro_emb/embeddings/train/interpro_emb_train_"+str(fold_idx)+".npy"
		train_ppi_mat = ont_sdir+"/data_splits/ppi_emb/transductive/train/ppi_emb_train_"+str(fold_idx)+".npy"
		train_uniprot_mat = ont_sdir+"/data_splits/uniprot_emb/train/uniprot_train_"+str(fold_idx)+".npy"

		test_interpro_mat = ont_sdir+"/data_splits/interpro_emb/embeddings/test/interpro_emb_test_"+str(fold_idx)+".npy"
		test_ppi_mat = ont_sdir+"/data_splits/ppi_emb/transductive/test/ppi_emb_test_"+str(fold_idx)+".npy"
		test_uniprot_mat = ont_sdir+"/data_splits/uniprot_emb/test/uniprot_test_"+str(fold_idx)+".npy"

	else:
		train_interpro_mat = ont_sdir+"/data_splits/interpro_emb/embeddings/train/interpro_emb_train_"+str(fold_idx)+".npy"
		train_ppi_mat = ont_sdir+"/data_splits/ppi_emb/inductive/train/ppi_emb_train_"+str(fold_idx)+".npy"
		train_uniprot_mat = ont_sdir+"/data_splits/uniprot_emb/train/uniprot_train_"+str(fold_idx)+".npy"

		test_interpro_mat = ont_sdir+"/data_splits/interpro_emb/embeddings/test/interpro_emb_test_"+str(fold_idx)+".npy"
		test_ppi_mat = ont_sdir+"/data_splits/ppi_emb/inductive/test/ppi_emb_test_"+str(fold_idx)+".npy"
		test_uniprot_mat = ont_sdir+"/data_splits/uniprot_emb/test/uniprot_test_"+str(fold_idx)+".npy"

	train_feature_list = [train_interpro_mat, train_ppi_mat, train_uniprot_mat]
	test_feature_list = [test_interpro_mat, test_ppi_mat, test_uniprot_mat]

	return train_feature_list, test_feature_list


def main(iworkmode=None):
	"""
	Task: to run the LTR solver vis the the ltr_wrapper
	"""

	ont_dirs = ["mf", "cc", "bp"]
	n_splits = 10

	ppi_modes = ["transductive"]#, "inductive"]
	feature_names = ["interpro", "ppi", "uniprot"]

	fold_i = sys.argv[1]

	for sdir in ont_dirs:
		print("Processing "+ sdir+ " ontology!")
		print("+"*85)
		###.......for fold_i in range(n_splits):
		y_train = np.load(sdir+"/data_splits/y/train/y_train_"+str(fold_i)+".npy")[:,1:].astype(int)
		y_test = np.load(sdir+"/data_splits/y/test/y_test_"+str(fold_i)+".npy")[:,1:].astype(int)
		
		train_feat_list, test_feat_list = create_feature_list(sdir, fold_i, ppi_mode)
		train_feat_combos = generate_multiview_configurations(train_feat_list)
		test_feat_combos = generate_multiview_configurations(test_feat_list)

		feature_names_combos = generate_multiview_configurations(feature_names)

		for i in range(len(train_feat_combos)):
			train_feat_views = train_feat_combos[i]
			test_feat_views = test_feat_combos[i]
			feature_names_list = feature_names_combos[i]

			nxfile = len(train_feat_views)
			lx_train = [np.load(train_feat_views[i])[:,:] for i in range(nxfile)]
			lx_test = [np.load(test_feat_views[i])[:,:] for i in range(nxfile)]

			lx_cent_train = [centr_kern(Ktrain=lx_train[i], Ktest=None, train=True) for i in range(nxfile)]
			lx_cent_test = [centr_kern(Ktrain=lx_train[i], Ktest=lx_test[i],train=False) for i in range(nxfile)]

			loss_types = 'huber_power'

			iter_count += 1
			print("Training with: ", loss_type, " loss function!")
			print("-"*85)
			closs = loss_diff_cls(stype=loss_type)

			parameters = { 'order': 10, 'rank': 1000, 'rankuv': 600, \
				'loss_diff': closs.loss_diff, 'nepoch' : 10 }    #1000, 600
			cmodel=ltr_cls(**parameters)
			
			lxtrain = lx_cent_train
			lxtest = lx_cent_test
			llinks = llinks_train = [[i] for i in range(len(train_feat_views))]

			cmodel.fit(lxtrain, y_train, llinks = llinks, xindex = None, nepoch=10)

			#Training
			print("Train results")
			Ypred = cmodel.predict(lxtrain, Ytrain = y_train, llinks = llinks, xindex = None)
			acc_eval(y_train[:,:], Ypred[:,:])
			print("*"*65)
			
			#Inference
			#Generating predictions based on raw scores only without selecting from the closest vector
			Ypred_ = cmodel.predict(lxtest, Ytrain = None, llinks = llinks, xindex = None)
			Ypred = expit(Ypred_)
			feat_names_concat = "_".join(feature_names_list)

			if "ppi" in feature_names_list:
				file_name_str = "ypred_"+feat_names_concat+"_"+ppi_mode+"_"+loss_type+"_"+str(fold_i)+".npy"
			else:
				file_name_str = "ypred_"+feat_names_concat+"_"+loss_type+"_"+str(fold_i)+".npy"
			if len(train_feat_views) == 1:
				file_name = sdir+"/test_predictions/ltr/single_view/"+file_name_str
				np.save(file_name, Ypred)
			elif len(train_feat_views) == 2:
				file_name = sdir+"/test_predictions/ltr/two_view/"+file_name_str
				np.save(file_name, Ypred)
			else:
				file_name =sdir+"/test_predictions/ltr/three_view/"+file_name_str
				np.save(file_name, Ypred)
			print("Finished saving results for: ",feat_names_concat, ", ", loss_type, "!")
			#***********************************************************************************************
			print("_"*85)
			print("_"*85)
			##time.sleep(10)


	#######################################################################################################################
	print('Bye!')
		##"""
	return(0)

# ## ###################################################
# ## ################################################################
if __name__ == "__main__":
	if len(sys.argv)==1:
		iworkmode=0
	elif len(sys.argv)>=2:
		iworkmode=eval(sys.argv[1])
	main(iworkmode)


