

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import scipy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import pathlib
import sklearn.metrics
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

import random
import time

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def model_outcomes_coal(model, coal, local_obs, ref_point, imp=None, predict_proba=False, seed=0):
    
    start = time.time()
    
    ## this function outputs a list of all model predictions for a single coalition
    # the input is the coalition of interest as a binary vector eg. [1,1,0,1]
    # ref point is the entire reference distirbution (here we take x_train)
    # local_obs is the instance we aim to explain
    
    n_obs, d_obs = np.shape(local_obs)

    # we consider all 2**n coalitions and all n reference points
    # get constants
    
    n, d = np.shape(ref_point)
    assert d == d_obs
    
    # train imputation algorithm for conditional references   
    # create an "all_imputed_coalitions" 3D matrix with all "artificial inputs" we get by imputing
    coalitions = np.array([[int(i) for i in '0'*((d)-len(bin(j))+2) + bin(j)[2:]] for j in range(2**d)]) 
    all_imputed_coalitions = np.zeros((n_obs, n, d))
    
    
    
    
    
     ## below we build the concatenated inputs using conditional imputation for dropped features and averages
    for k in range(n):#reference point index
        vect=(1-coal) 
        vect = vect.astype('float')
        vect[vect == 1] = 'nan'    # vect is a binary vector of being "absent" ('nan') or "present" (1)
        # impute conditionally with Bayesian Ridge MICE 
        imputed_coalitions = coal * local_obs + vect #either local obs value or 'nan'
        #print(imputed_coalitions)
        nans = (np.isnan(imputed_coalitions).sum(axis=0) > 0)
        if (~nans).sum() == 0 or imp is None:
            idx = np.random.randint(n, size=n_obs)
            imputation = ref_point[idx]
            imputed_coalitions[:,nans] = imputation[:,nans]
        else:
            imputed_coalitions = imp.transform(imputed_coalitions)       
        all_imputed_coalitions[:, k, :] = imputed_coalitions
        
    end = time.time()
    print('Execution time (s) :', end - start)
    #we ultimately return the model predictions
    if predict_proba:
        return model.predict_proba(all_imputed_coalitions.reshape(n_obs * n, d))[:,1].reshape(n_obs, n)
    else:
        return model.predict(all_imputed_coalitions.reshape(n_obs * n, d)).reshape(n_obs, n)


class IterativeImputerSubsets(object):

    def __init__(self, subsets=[], **kwargs):

        self.imps = []
        self.subsets = subsets
        for subset in self.subsets:
            self.imps.append(IterativeImputer(**kwargs))

    def fit(self, x_train):
        for imp, subset in zip(self.imps, self.subsets):
            x_train_subset = x_train[:, subset]
            imp.fit(x_train_subset)

    def transform(self, x):
        x_new = x
        for imp, subset in zip(self.imps, self.subsets):
            x_new[:, subset] = imp.transform(x[:, subset])
        return x_new

psis_specific_q_ratios = []
psis_specific_d_ratios = []
phi_direct_cs_ratios = []
phi_indirect_cs_ratios = []


for iteration in range(25):

    start = time.time()

    N = 200
    p = 0.5 * np.ones(N)
    t = np.random.binomial(n=1, p=p)

    q = np.random.uniform(size=N)
    d = np.random.binomial(n=1, p=4 / 5 - 3 / 5 * t)

    loc=5
    scale=10
    (coef_t, coef_q, coef_d, coef_t_q, coef_t_d) = tuple(np.random.normal(loc=loc, scale=scale, size=5))
    y = coef_t*t + coef_q*q + coef_d*d + coef_t_q*t*q + coef_t_d*t*d


    x = pd.DataFrame({'q': q, 'd': d, 't': t})
    prop_train = 0.5
    N_train = int(prop_train * N)
    x_train = x.iloc[:N_train]
    y_train = y[:N_train]
    x_test = x.iloc[N_train:]
    y_test = y[N_train:]
    imp = IterativeImputer(max_iter=100, random_state=0, sample_posterior=True)
    imp.fit(x_train.values) #imputer learns from marginal distribution



    imp_all_but_t = IterativeImputer(max_iter=100, random_state=0, sample_posterior=True)
    imp_all_but_t.fit(x_train[['q','d']].values)


    outcome_poly = LinearRegression()
    poly = PolynomialFeatures(2)
    outcome = Pipeline([('poly',poly),('outcome_poly',outcome_poly)])
    outcome.fit(x_train, y_train)
    propensity = LogisticRegression()
    propensity.fit(x_train[['q','d']], x_train['t'])



    print('outcomes_all')
    outcomes_all = model_outcomes_coal(model=outcome, coal=np.array([1,1,1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp)
    print('outcomes_all_but_t')
    outcomes_all_but_t = model_outcomes_coal(model=outcome, coal=np.array([1,1,0]), local_obs=x_test.values, ref_point=x_train.values, imp=imp)
    print('propensities_all')
    propensities_all =  model_outcomes_coal(model=propensity, coal=np.array([1,1]), local_obs=x_test[['q','d']].values, ref_point=x_train[['q','d']].values, imp=imp_all_but_t, predict_proba=True)

    psis_coal_all = (outcomes_all.mean(axis=1) - outcomes_all_but_t.mean(axis=1)) / (x_test['t'].values - propensities_all.mean(axis=1))


    print('outcomes_all_but_q')
    outcomes_all_but_q = model_outcomes_coal(model=outcome, coal=np.array([0,1,1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp)
    print('outcomes_all_but_q_t')
    outcomes_all_but_q_t = model_outcomes_coal(model=outcome, coal=np.array([0,1,0]), local_obs=x_test.values, ref_point=x_train.values, imp=imp)
    print('propensities_all_but_q')
    propensities_all_but_q =  model_outcomes_coal(model=propensity, coal=np.array([0,1]), local_obs=x_test[['q','d']].values, ref_point=x_train[['q','d']].values, imp=imp_all_but_t, predict_proba=True)

    psis_coal_all_but_q = (outcomes_all_but_q.mean(axis=1) - outcomes_all_but_q_t.mean(axis=1)) / (x_test['t'].values - propensities_all_but_q.mean(axis=1))

    psis_specific_q = psis_coal_all - psis_coal_all_but_q
    print('psis_specific_q_ratio: ', np.abs(psis_specific_q).mean() / y_train.std())
    psis_specific_q_ratios.append(np.abs(psis_specific_q.mean()) / y_train.std())



    print('outcomes_all_but_d')
    outcomes_all_but_d = model_outcomes_coal(model=outcome, coal=np.array([1,0,1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp)
    print('outcomes_all_but_d_t')
    outcomes_all_but_d_t = model_outcomes_coal(model=outcome, coal=np.array([1,0,0]), local_obs=x_test.values, ref_point=x_train.values, imp=imp)
    print('propensities_all_but_d')
    propensities_all_but_d =  model_outcomes_coal(model=propensity, coal=np.array([1,0]), local_obs=x_test[['q','d']].values, ref_point=x_train[['q','d']].values, imp=imp_all_but_t, predict_proba=True)


    psis_coal_all_but_d = (outcomes_all_but_d.mean(axis=1) - outcomes_all_but_d_t.mean(axis=1)) / (x_test['t'].values - propensities_all_but_d.mean(axis=1))

    psis_specific_d = psis_coal_all - psis_coal_all_but_d
    print('psis_specific_d_ratio: ', np.abs(psis_specific_d).mean() / y_train.std())
    psis_specific_d_ratios.append(np.abs(psis_specific_d.mean()) / y_train.std())




    imp_without_t = IterativeImputerSubsets([[True,True,False]], max_iter=100, random_state=0, sample_posterior=True)
    imp_without_t.fit(x_train.values) #imputer learns from marginal distribution


    # In[30]:


    # decorrelate a subset of features from others (but not between features in said subset)
    def decorrelate_x_train(x_train, mask_to_decorrelate):
        x_train_decorrelated = x_train
        idx = np.random.randint(len(x_train), size=len(x_train))
        x_train_sampled = x_train[idx]
        x_train_decorrelated[:,mask_to_decorrelate] = x_train_sampled[:,mask_to_decorrelate]
        return x_train_decorrelated


    # In[31]:


    x_train_do_q = decorrelate_x_train(x_train.values, [True,False,False])
    imp_do_q = IterativeImputer(max_iter=100, random_state=0, sample_posterior=True)
    imp_do_q.fit(x_train_do_q)
    imp_do_q_without_t =  IterativeImputerSubsets([[True,True,False]], max_iter=100, random_state=0, sample_posterior=True)
    imp_do_q_without_t.fit(x_train_do_q)


    # In[32]:


    x_train_do_d = decorrelate_x_train(x_train.values, [True,False,False])
    imp_do_d = IterativeImputer(max_iter=100, random_state=0, sample_posterior=True)
    imp_do_d.fit(x_train_do_d)
    imp_do_d_without_t =  IterativeImputerSubsets([[True,True,False]], max_iter=100, random_state=0, sample_posterior=True)
    imp_do_d_without_t.fit(x_train_do_d)


    # Causal Shapley


    # Full coalition - the distribution becomes independent
    model_outcome_all_cs = np.mean(model_outcomes_coal(outcome, np.array([1, 1, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=None))
    model_outcome_all_but_t_cs = np.mean(model_outcomes_coal(outcome, np.array([1, 1, 0]), local_obs=x_test.values, ref_point=x_train.values, imp=None))
    model_outcome_all_impwithoutt_cs = np.mean(model_outcomes_coal(outcome, np.array([1, 1, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=None))

    phi_direct_cs_full_coalition = model_outcome_all_impwithoutt_cs - model_outcome_all_but_t_cs
    phi_indirect_cs_full_coalition = model_outcome_all_cs - model_outcome_all_impwithoutt_cs




    # Q coalition
    model_outcome_all_but_d_cs = np.mean(model_outcomes_coal(outcome, np.array([1, 0, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_do_q))
    model_outcome_all_but_d_t_cs = np.mean(model_outcomes_coal(outcome, np.array([1, 0, 0]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_do_q))
    model_outcome_all_but_d_impwithoutt_cs = np.mean(model_outcomes_coal(outcome, np.array([1, 0, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_do_q_without_t))

    phi_direct_cs_q_coalition = model_outcome_all_but_d_impwithoutt_cs - model_outcome_all_but_d_t_cs
    phi_indirect_cs_q_coalition = model_outcome_all_but_d_cs - model_outcome_all_but_d_impwithoutt_cs

    # D coalition
    model_outcome_all_but_q_cs = np.mean(model_outcomes_coal(outcome, np.array([0, 1, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_do_d))
    model_outcome_all_but_q_t_cs = np.mean(model_outcomes_coal(outcome, np.array([0, 1, 0]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_do_d))
    model_outcome_all_but_q_impwithoutt_cs = np.mean(model_outcomes_coal(outcome, np.array([0, 1, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_do_d_without_t))

    phi_direct_cs_d_coalition = model_outcome_all_but_q_impwithoutt_cs - model_outcome_all_but_q_t_cs
    phi_indirect_cs_d_coalition = model_outcome_all_but_q_cs - model_outcome_all_but_q_impwithoutt_cs



    # Empty coalition
    model_outcome_all_but_q_d_cs = np.mean(model_outcomes_coal(outcome, np.array([0, 0, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp))
    model_outcome_all_but_q_d_t_cs = np.mean(model_outcomes_coal(outcome, np.array([0, 0, 0]), local_obs=x_test.values, ref_point=x_train.values, imp=imp))
    model_outcome_all_but_q_d_impwithoutt_cs = np.mean(model_outcomes_coal(outcome, np.array([0, 0, 1]), local_obs=x_test.values, ref_point=x_train.values, imp=imp_without_t))

    phi_direct_cs_empty_coalition = model_outcome_all_but_q_d_impwithoutt_cs - model_outcome_all_but_q_d_t_cs
    phi_indirect_cs_empty_coalition = model_outcome_all_but_q_d_cs - model_outcome_all_but_q_d_impwithoutt_cs


    phi_direct_cs = 1/3*phi_direct_cs_full_coalition + 1/6*phi_direct_cs_q_coalition + 1/6*phi_direct_cs_d_coalition + 1/3*phi_direct_cs_empty_coalition
    phi_indirect_cs = 1/3*phi_indirect_cs_full_coalition + 1/6*phi_indirect_cs_q_coalition + 1/6*phi_indirect_cs_d_coalition + 1/3*phi_indirect_cs_empty_coalition

    print('phi direct CS / sigma :', phi_direct_cs / y_train.std())
    print('phi indirect CS / sigma :', phi_indirect_cs / y_train.std())

    phi_direct_cs_ratios.append(np.abs(phi_direct_cs) / y_train.std())
    phi_indirect_cs_ratios.append(np.abs(phi_indirect_cs) / y_train.std())

    end = time.time()
    print(f'TOTAL TIME FOR ITERATION {iteration} : {end - start}')

df = dict(
    psis_specific_q_ratios = psis_specific_q_ratios,
    psis_specific_d_ratios = psis_specific_d_ratios,
    phi_direct_cs_ratios = phi_direct_cs_ratios,
    phi_indirect_cs_ratios = phi_indirect_cs_ratios
)
df = pd.DataFrame(df)

df.to_csv('../outputs/mediation_one_mediator.csv')