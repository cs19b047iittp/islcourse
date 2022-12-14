# kali
import torch
from torch import nn
import torch.optim as optim

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######
import sklearn.cluster as skl_cluster
import sklearn.datasets as skl_data
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score

def get_data_blobs(n_points=100):
  X, y = skl_data.make_blobs(n_samples=n_points, cluster_std=0.8, centers=3, random_state=47)
  return X,y

def get_data_circles(n_points=100):
  X, y = skl_data.make_circles(n_samples=n_points, noise=.07, random_state=47)
  return X,y

def get_data_mnist():
  X,y = load_digits(return_X_y=True)
  return X,y

def build_kmeans(X=None,k=10):
  km = skl_cluster.KMeans(n_clusters=k)
  km.fit(X)
  return km

def assign_kmeans(km=None,X=None):
  ypred = None
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h=homogeneity_score(ypred_1,ypred_2)
  c=completeness_score(ypred_1,ypred_2)
  v=v_measure_score(ypred_1,ypred_2)
  return h,c,v

###### PART 2 ######

import numpy as np
from sklearn.datasets import load_digits

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn import metrics

def build_lr_model(X=None, y=None):
  lr_model = LogisticRegression(solver='liblinear', random_state=47)
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  rf_model=RandomForestClassifier(n_estimators=100, rando_state=47)
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  pred=model1.predict(X)
  acc=accuracy_score(y, pred)
  precision,recall,fscore,support=score(y,pred,average='macro')
  f_pr, t_pr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
  auc=metrics.auc(f_pr, t_pr)
  # write your code here...
  return acc, precision, recall, fscore, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {'solver':['liblinear'], "C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = {
        "max_depth": [1, 10, None],
        "criterion": ['gini', 'entropy'],
        "n_estimators": [1, 10, 100]
    }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  grid_search_cv = GridSearchCV(model1, param_grid, cv = cv, scoring=metrics, refit=False)
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  grid_search_cv.fit(X, y)
  # pred = grid_search_cv.predict(X)

  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  # print(grid_search_cv.cv_results_)
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list

  top1_scores = []
  for metric in metrics:
      top1_scores.append(grid_search_cv.cv_results_[metric][grid_search_cv.best_index_])

  return top1_scores

###### PART 3 ######

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self)
    
    self.fc_encoder = None # write your code inp_dim to hid_dim mapper
    self.fc_decoder = None # write your code hid_dim to inp_dim mapper
    self.fc_classifier = None # write your code to map hid_dim to num_classes
    
    self.relu = None #write your code - relu object
    self.softmax = None #write your code - softmax object
    
  def forward(self,x):
    x = None # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    lc1 = None # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X, y = None, None
  # write your code
  return X,y

# def get_loss_on_single_point(mynn=None,x0,y0):
#   y_pred, xencdec = mynn(x0)
#   lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
#   # the lossval should have grad_fn attribute set
#   return lossval

# def train_combined_encdec_predictor(mynn=None,X,y, epochs=11):
#   # X, y are provided as tensor
#   # perform training on the entire data set (no batches etc.)
#   # for each epoch, update weights
  
#   optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
#   for i in range(epochs):
#     optimizer.zero_grad()
#     ypred, Xencdec = mynn(X)
#     lval = mynn.loss_fn(X,y,ypred,Xencdec)
#     lval.backward()
#     optimzer.step()
    
#   return mynn
