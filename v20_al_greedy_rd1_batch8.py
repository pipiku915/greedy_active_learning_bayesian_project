# %%
import jax.numpy as jnp
import jax
from jax import random, vmap
from scipy.special import logit, expit
from jax import jit
import os
import tqdm
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2" 
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import random as rd

import numpyro
numpyro.set_platform('gpu')
# numpyro.set_platform('cpu')
print('Jax Version:', jax.__version__) 
print('Numpyro Version: ', numpyro.__version__) 
print(jax.config.FLAGS.jax_backend_target) 
print(jax.lib.xla_bridge.get_backend().platform) 
n_parallel = jax.local_device_count()
print('Number of compuation devices', n_parallel)


import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs,Predictive,HMCGibbs, SVI, Trace_ELBO, autoguide

from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect

from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoNormal
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect
from numpyro import handlers
from sklearn.metrics import mean_squared_error


import numpy as np
import sys
import pickle

from scipy import stats
from torch import reshape

import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

import time as time


# from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
import scipy.sparse as sp
from typing import List, Sequence, Union
import warnings
import torch
from pynvml import *
import gc

import kcenter_greedy as kg

# %% 
# 
### 0. function related to import multiple input datasets
def input_data_process(main_tensor_dir, visual_face_dir, language_trait_dir, subset_size, train_size):
    print('Loading main tensor dataset ......')
    ### Load main tensor dataset
    rating_tensor_df = pd.read_csv(main_tensor_dir, low_memory = False)
    # print(rating_tensor_df.describe())
    # print(rating_tensor_df.head(3))
    # print('------------------------------------------------------------------------')
    ### trim the faces with no rating data
    rating_tensor_df = rating_tensor_df[rating_tensor_df["stimulus"] <= 1000]

    # normalize participant index over the whole dataset
    rating_tensor_df['participant'] = np.unique(rating_tensor_df['participant'], return_inverse=True)[1]  


    ### rescale the ratings to aviod infinite values
    rating_tensor_df['response'] = rating_tensor_df['response'] / 100.0
    rating_tensor_df['response'] = rating_tensor_df['response'].replace([0], 0.01)
    rating_tensor_df['response'] = rating_tensor_df['response'].replace([1], 0.99)
    # print(rating_tensor_df.head(3))
    # print(rating_tensor_df.describe())
    print('!!!! Total number of records before sampling: ', rating_tensor_df.shape)
    rating_all = rating_tensor_df.to_numpy()


    ### begin to sample training and test datasets
    rating_tensor_df_sampled = rating_tensor_df.sample(n = subset_size, replace = False).reset_index(drop = True)
    rating_tensor_df_sampled['participant'] = np.unique(rating_tensor_df_sampled['participant'], return_inverse=True)[1]  # normalize index
    # print(rating_tensor_df_sampled.head(5))
    rating_training_df = rating_tensor_df_sampled.sample(n = train_size, replace = False).reset_index(drop = True)
    rating_test_df = rating_tensor_df_sampled[~rating_tensor_df_sampled.index.isin(rating_training_df.index)].reset_index(drop = True)
    rating_training = rating_training_df.to_numpy()
    # rating_training[:, 3] = rating_training[:, 3].astype(np.float)
    rating_test = rating_test_df.to_numpy()

    # # print(rating_training_df.head(5))
    print('rating all shape: ', rating_all.shape)
    print('rating train shape: ', rating_training.shape)
    print('rating test shape: ', rating_test.shape)

    ### Load side feature datasets
    # print('Loading side feature datasets ......')
    visual_face_vectors = np.load(visual_face_dir,allow_pickle=True)
    language_trait_vectors = np.load(language_trait_dir, allow_pickle=True)
    visual_face_vectors = jnp.array(visual_face_vectors)
    language_trait_vectors = jnp.array(language_trait_vectors)

    return rating_training, rating_test, rating_all, visual_face_vectors, language_trait_vectors

# %%
def model(features, participants, stimulus, traits, latent_dimensions, rates = None):

    grandMean = numpyro.sample("grandMean", dist.Normal(0.0, 1.0))
    # print('geandMean: ', grandMean)

    bias_prior, scale_prior = numpyro.sample("bias_scaling_Prior", dist.Gamma(1.0, 10.0).expand((2,)))
    # with numpyro.plate('scaling_bias_plate', participants):
    with numpyro.plate('scaling_bias_plate', 4476):
        bias = numpyro.sample("bias", dist.Normal(0, bias_prior))
        # print('Bias: ', bias)
        # print('Bias shape: ', bias.shape)
        scaling = numpyro.sample("scale", dist.Normal(1, scale_prior))
        # print('Scaling: ', scaling)


    ###bayesian tensor factorization
    visual_f_prior = numpyro.sample("visual_f_prior", dist.Gamma(10.0, 1.0))
    with numpyro.plate('latent_visual_coefficients', 512):
        # with numpyro.plate('visual_dimensions', latent_dimensions):
        with numpyro.plate('visual_dimensions', 10):
            visual_feature_coefficient =  numpyro.sample("visual_feature_coefficient",dist.Normal(0, 1/visual_f_prior))
            # print(visual_feature_coefficient.shape)
            ### (512 latent vector weights, 10 latent spaces)

    language_f_prior = numpyro.sample("language_f_prior", dist.Gamma(10.0, 1.0))
    with numpyro.plate('latent_language_coefficients', 300):
        # with numpyro.plate('language_dimensions', latent_dimensions):
        with numpyro.plate('language_dimensions', 10):
            language_feature_coefficient =  numpyro.sample("language_feature_coefficient", dist.Normal(0, 1/language_f_prior))
            # print(language_feature_coefficient.reshape(300, 10))
            # print(language_feature_coefficient.shape)

    mu_part1 = vmap(lambda vis_vec, vis_coefs:
                    jnp.dot(vis_vec, vis_coefs.T), in_axes = (0, None))(
                    visual_face_vectors[features[:, 1] - 1],
                    visual_feature_coefficient
                    )
    # print('mu_part1: ', mu_part1.shape)

    mu_part2 = vmap(lambda lang_vec, lang_coefs:
                    jnp.dot(lang_vec, lang_coefs.T), in_axes = (0, None))(
                    language_trait_vectors[features[:, 2] - 1],
                    language_feature_coefficient
                    )
    # print('mu_part2: ', mu_part2.shape)

    mu = vmap(lambda p1, p2, b, scale: (jnp.log(scale) * (grandMean + jnp.vdot(p1, p2))) + b)(mu_part1, 
                                                                                           mu_part2, 
                                                                                           bias[features[:, 0]],
                                                                                           scaling[features[:, 0]]
                                                                                        )

    # print('!!!!!!!!!!!!!!!mu final minimum: ', jnp.min(mu))
    # print('!!!!!!!!!!!!!!!mu final maximum: ', jnp.max(mu))
    # mu_test = vmap(lambda p1, p2: jnp.vdot(p1, p2) )(mu_part1, mu_part2)
    # # print('mu_combined: ',mu_test)
    # print('!!!!!!!!!!!!!!!mu_test shape: ', mu_test.shape)
    # print('!!!!!!!!!!!!!!!mu_test minimum: ', jnp.min(mu_test))
    # print('!!!!!!!!!!!!!!!mu_test maximum: ', jnp.max(mu_test))
    


    sigma = numpyro.sample("sigma_val", dist.HalfNormal(4.0))
    # print('sigma val: ', sigma)
    numpyro.sample("rating", dist.Normal(mu, 1/sigma), obs = rates)

# %%
def predict_fn_jit(rng_key, samples, *args, **kwargs):
    return Predictive(model, samples, parallel=True)(rng_key, *args, **kwargs)

# %%
def bayesian_mcmc_run_predict(model_input, 
                            features, 
                            participants, 
                            stimulus , 
                            traits , 
                            latent_dimensions, 
                            rates,
                            features_test, 
                            participants_test, 
                            stimulus_test, 
                            traits_test, 
                            latent_dimensions_test
                            ):
    used_GPU0_lst =[]
    used_GPU1_lst =[]
    used_GPU2_lst =[]
    start_time = time.time()

    kernel = NUTS(model_input, init_strategy=numpyro.infer.init_to_sample, step_size=1e-3, max_tree_depth=6)
    mcmc = MCMC(kernel, num_warmup=100, num_samples=10, chain_method='parallel', num_chains= 1, progress_bar=False, jit_model_args=True)
    # kernel = NUTS(model_input)
    # mcmc = MCMC(kernel, num_warmup=100, num_samples=20, num_chains= 1, progress_bar=False, jit_model_args=True)
    
    print('inital GPU cost:---------------------------------------------------------')
    info_gpu0_mb_take0, info_gpu1_mb_take0, info_gpu2_mb_take0 = GPU_mem_state()
    used_GPU0_lst.append(info_gpu0_mb_take0)
    used_GPU1_lst.append(info_gpu0_mb_take0)
    used_GPU2_lst.append(info_gpu2_mb_take0)
    print('--------------------------------------------------------------------------')
    N_runs = 8
    # for i in tqdm.tqdm(range(N_runs)):
    for i in range(N_runs):
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        # info_gpu0_mb_take, info_gpu1_mb_take, info_gpu2_mb_take = GPU_mem_state()
        # used_GPU0_lst.append(info_gpu0_mb_take)
        # used_GPU1_lst.append(info_gpu1_mb_take)
        # used_GPU2_lst.append(info_gpu2_mb_take)

    
        mcmc.run(random.PRNGKey(i), features, participants, stimulus, traits, latent_dimensions, rates)
        # posterior_samples = mcmc.get_samples()
        if i == 7:
            posterior_samples = mcmc.get_samples()
            # posterior_trace = posterior_samples            
            # posterior_trace = [np.atleast_1d(np.asarray(f)) for f in posterior_samples]
            predictions_jit = predict_fn_jit(random.PRNGKey(i), posterior_samples, features, participants, stimulus, traits, latent_dimensions)["rating"]
            test_predictions_jit = predict_fn_jit(random.PRNGKey(i), posterior_samples, features_test, participants_test, stimulus_test, traits_test, latent_dimensions_test)["rating"]
            del posterior_samples
            gc.collect()

        
        del os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]
        # del posterior_samples
        mcmc._warmup_state = mcmc._last_state
        gc.collect()

    

    ### compute training prediction
    model_pred_rating_array = np.array(predictions_jit)
    print('Shape of original training predicted rating arrays', model_pred_rating_array.shape)
    # print('Type of original predicted rating arrays', type(model_pred_rating_array))

    pred_ratings = np.round(np.nanmean((expit(model_pred_rating_array) * 100),axis = 0), decimals = 4)
    # print('Shape of training predicted rating values', pred_ratings.shape)
    # print('Training predicted rating values', pred_ratings)
    pred_ratings_std = np.round(np.nanstd((expit(model_pred_rating_array) * 100),axis = 0), decimals = 4)
    # print('Shape of training predicted rating std', pred_ratings_std.shape)

    ### compute test prediction
    test_model_pred_rating_array = np.array(test_predictions_jit)

    # print('Shape of original test predicted rating arrays', test_model_pred_rating_array.shape)
    test_pred_ratings = np.round(np.nanmean((expit(test_model_pred_rating_array) * 100),axis = 0), decimals = 4)
    test_pred_ratings[np.isnan(test_pred_ratings)] = 0
    # print('Shape of test predicted rating values', test_pred_ratings.shape)
    # print('Test predicted rating values', test_pred_ratings)
    # print('Type of Test predicted rating values', type(test_pred_ratings))
    test_pred_ratings_std = np.round(np.nanstd((expit(test_model_pred_rating_array) * 100),axis = 0), decimals = 4)
    test_pred_ratings_std[np.isnan(test_pred_ratings_std)] = 0
    # print('Shape of test predicted rating std', test_pred_ratings_std.shape)

    del predictions_jit, test_predictions_jit, mcmc, kernel
    gc.collect()

    print('ended GPU cost:---------------------------------------------------------')
    info_gpu0_mb_take2, info_gpu1_mb_take2, info_gpu2_mb_take2 = GPU_mem_state()
    used_GPU0_lst.append(info_gpu0_mb_take2)
    used_GPU1_lst.append(info_gpu1_mb_take2)
    used_GPU2_lst.append(info_gpu2_mb_take2)
    print('--------------------------------------------------------------------------')

    print("------------- %s seconds simulation time -----------------" % (time.time() - start_time))
    # return posterior_trace, pred_ratings, pred_ratings_std, test_pred_ratings, test_pred_ratings_std, used_GPU0_lst, used_GPU2_lst
    return pred_ratings, pred_ratings_std, test_pred_ratings, test_pred_ratings_std, used_GPU0_lst, used_GPU1_lst, used_GPU2_lst
# %%
def prediction_outtable_df(data0 , pred_ratings, pred_ratings_std):

    participant_lst = list(data0[:, 0]) ### stimulus list -- no unique
    # print(len(participant_lst))
    stimulus_lst = list(data0[:, 1]) ### stimulus list -- no unique
    # print(len(stimulus_lst))
    trait_lst = list(data0[:, 2]) ### trait list -- no unqiue
    # print(len(trait_lst))
    rating_lst = list(data0[:, 3] * 100) ### trait list -- no unqiue
    # print(len(rating_lst))

    prediction_out_df = pd.DataFrame(columns=['participant', 'stimulus', 'trait', 'rating_true', 'rating_pred', 'rating_std'])
    prediction_out_df['participant'] = participant_lst
    prediction_out_df['stimulus'] = stimulus_lst
    prediction_out_df['trait'] = trait_lst
    prediction_out_df['rating_true'] = rating_lst
    prediction_out_df['rating_pred'] = pred_ratings
    prediction_out_df['rating_std'] = pred_ratings_std

    prediction_out_df['rating_pred_inf'] = prediction_out_df['rating_pred'] - prediction_out_df['rating_std']
    prediction_out_df['rating_pred_sup'] = prediction_out_df['rating_pred'] + prediction_out_df['rating_std']

    # print(prediction_out_df.shape)
    # print(prediction_out_df.head(5))
    # print(prediction_out_df.describe())

    return prediction_out_df
# %%
def RMSE_val(pred_ratings, actual_ratings):
    # RMSE_value = mean_squared_error(prediction_out_df['rating_true'], prediction_out_df['rating_pred'], squared=False)
    RMSE_value = mean_squared_error(pred_ratings, actual_ratings, squared=False)
    return RMSE_value


# %%
### Plot gpu0 usage while running
def plot_gpu_usage(GPU_avaiable_mb_lst, save_path):
    # summarize history for RMSE
    plt.plot(range(1, len(GPU_avaiable_mb_lst)+1), GPU_avaiable_mb_lst)
    plt.title('Free GPU usage while running')
    plt.ylabel('usage in MB')
    plt.xlabel('each run')
    plt.legend(['MB'], loc='lower right')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()
    plt.close()

# %%
# %%
### Option 3: greedy batch selection:

def kgreedy_aggree_TopN(X, new_test_ratings, new_test_ratings_std, select_batch_func, already_selected, top_N = 1):
    start_time2 = time.time()

    selected_new_batch = select_batch_func(already_selected, top_N)
    selected_new_batch = np.array(selected_new_batch)
    print('Indexes of selected new batch: ', selected_new_batch)
    # print('Already selected 3: ', already_selected)
    query_idx = selected_new_batch

    queried_X_row = np.asarray(X[selected_new_batch,:])
    # queried_y_row = np.asarray(y[selected_new_batch])

    X_new = np.delete(X, selected_new_batch, axis = 0)
    # y_new = jnp.delete(y, selected_new_batch, axis = 0)

    print("------------- %s seconds sampling time -----------------" % (time.time() - start_time2))

    return query_idx, queried_X_row, X_new

# def GP_regression_std_TopN(X, new_test_ratings, new_test_ratings_std, top_N = 1):
#     start_time_6 = time.time()
#     print('!!!!!!!!!!!!!! Test of how many std ', new_test_ratings_std.shape)
#     query_idx = np.argpartition(np.array(new_test_ratings_std), -1 *top_N)[-1*top_N:]
#     queried_X_row = np.asarray(X[query_idx,:])
#     X_new = np.delete(X, query_idx, axis = 0)
#     print('!!!!!!!!!maximum std: ', new_test_ratings_std[query_idx])
#     # print('!!!!!!!!!Maximum std rows in test data: ', queried_X_row)
#     # print('!!!!!!!!!Its predicted rating: ', new_test_ratings[query_idx])
#     print('!!!!!!!! Test Data shape after removed the selected point: ', X_new.shape)

#     print("------------- %s seconds greedy selection time -----------------" % (time.time() - start_time_6))

#     return query_idx, queried_X_row, X_new
# %%
### Model query functions
#### 1. Get rows of corresponding indexes
def retrieve_rows(X, I):
    """
    Returns the rows I from the data set X
    For a single index, the result is as follows:
    * 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame
    * row in case of list or numpy format
    """

    try:
        return X[I]
    except:
        if sp.issparse(X):
            # Out of the sparse matrix formats (sp.csc_matrix, sp.csr_matrix, sp.bsr_matrix,
            # sp.lil_matrix, sp.dok_matrix, sp.coo_matrix, sp.dia_matrix), only sp.bsr_matrix, sp.coo_matrix
            # and sp.dia_matrix don't support indexing and need to be converted to a sparse format
            # that does support indexing. It seems conversion to CSR is currently most efficient.

            sp_format = X.getformat()
            return X.tocsr()[I].asformat(sp_format)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[I]
        elif isinstance(X, list):
            return np.array(X)[I].tolist()
        elif isinstance(X, dict):
            X_return = {}
            for key, value in X.items():
                X_return[key] = retrieve_rows(value, I)
            return X_return

    raise TypeError("%s datatype is not supported" % type(X))


# %%
### 2.query data points from unlabeled data pool (in test)
def query(X_pool, selection_strategy, return_metrics = False):
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.
        Args:
            X_pool: Pool of unlabeled instances to retrieve most informative instances from
            return_metrics: boolean to indicate, if the corresponding query metrics should be (not) returned
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~modAL.uncertainty.uncertainty_sampling`, it is the pool of samples from which the query strategy
                should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.
        Returns:
            Value of the query_strategy function. Should be the indices of the instances from the pool chosen to be
            labelled and the instances themselves. Can be different in other cases, for instance only the instance to be
            labelled upon query synthesis.
            query_metrics: returns also the corresponding metrics, if return_metrics == True
        """

        try:
            query_idx, queried_X_pool_row, query_metrics = selection_strategy(X_pool)

        except:
            query_metrics = None
            query_idx, queried_X_pool_row = selection_strategy(X_pool)

        if return_metrics:
            if query_metrics is None: 
                warnings.warn("The selected query strategy doesn't support return_metrics")
            return query_idx, retrieve_rows(X_pool, query_idx), query_metrics
        else:
            return query_idx, retrieve_rows(X_pool, query_idx)


# %%
### 3. Two ways of concatenate newly generated data arrays with old one


def data_vstack(blocks):
    """
    Stack vertically sparse/dense arrays and pandas data frames.
    Args:
        blocks: Sequence of modALinput objects.
    Returns:
        New sequence of vertically stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.vstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        return blocks[0].append(blocks[1:])
    elif isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        return np.concatenate(blocks).tolist()

    try:
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks)
    except:
        pass

    raise TypeError("%s datatype is not supported" % type(blocks[0]))


def data_hstack(blocks):
    """
    Stack horizontally sparse/dense arrays and pandas data frames.
    Args:
        blocks: Sequence of modALinput objects.
    Returns:
        New sequence of horizontally stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        pd.concat(blocks, axis=1)
    elif isinstance(blocks[0], np.ndarray):
        return np.hstack(blocks)
    elif isinstance(blocks[0], list):
        return np.hstack(blocks).tolist()

    try:
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks, dim=1)
    except:
        pass

    TypeError("%s datatype is not supported" % type(blocks[0]))

# %%
### 4. Adds the new data and label to the known data, but does not retrain the model.
def _add_training_data(org_X, org_y, X, y):
        """
        Adds the new data and label to the known data, but does not retrain the model.
        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
        Note:
            If the classifier has been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        # check_X_y(X, y, accept_sparse=True, ensure_2d=False, allow_nd=True, multi_output=True, dtype=None,
        #           force_all_finite =True)
        X_training = org_X
        y_training = org_y

        if X_training is None:
            X_training = X
            y_training = y
        else:
            try:
                print('Comparison of shapes on old and new')
                # print(X_training.shape)
                # print(X.shape)
                # print(y_training.shape)
                # print(y.shape)
                X_training = data_vstack((X_training, X))
                y_training = data_vstack((y_training, y))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')
        X_training = np.array(X_training)
        y_training = np.array(y_training)
        # print('After query, what is new labelled training data shape: ', X_training.shape)
        # print('After query, what is new labelled training data type: ', type(X_training))
        # print('After query, what is new labelled training label shape: ', y_training.shape)
        # print('After query, what is new labelled training label type: ', type(y_training))
        # print(y_training)
        ### returns the newly generated training data and their true labels
        return X_training, y_training

# %%
### 6. Fits model to the training data and labels provided to it so far.
def fit_to_mcmc(X, y, X_test_update):
        """
        This function is to fit in the new training after adding 1 or 1 batch of queried points.

        """
        ### new training data  after added
        X = X.astype(np.int32)
        logit_y = logit(y) 
        # print(X)
        # print(logit_y)

        ### new test data after removed
        print('Check new test set:~~~~~~~~~~~~~~~~~~')
        print(X_test_update.shape)
        # print(X_test_update[0:5, :])
        X_new_test_update = X_test_update[:, 0:3].astype(np.int32)
        logit_y_new_test = logit(X_test_update[3]) 


        ### if choose mcmc simulation for new round
        new_train_pred_ratings, new_train_pred_ratings_std, new_test_pred_ratings, new_test_pred_ratings_std, new_used_GPU0_lst, new_used_GPU1_lst, new_used_GPU2_lst = bayesian_mcmc_run_predict(model_input = model,
                                                                                                                                                                            features=X,
                                                                                                                                                                            participants = all_participants,
                                                                                                                                                                            stimulus= all_stimulus,
                                                                                                                                                                            traits= all_traits,
                                                                                                                                                                            latent_dimensions=latent_dim,
                                                                                                                                                                            rates= logit_y,
                                                                                                                                                                            features_test =X_new_test_update,
                                                                                                                                                                            participants_test = all_participants,
                                                                                                                                                                            stimulus_test = all_stimulus,
                                                                                                                                                                            traits_test = all_traits,
                                                                                                                                                                            latent_dimensions_test =latent_dim
                                                                                                                                                                            )


        ########################################
        ### output new training predictions
        # print('New Training Prediction: ................')
        y_new_training_scaled = y * 100.0
        train_DATA_new = data_hstack([X, y.reshape(-1, 1)])
        # train_DATA_new2 = data_hstack([X, y_new_training_scaled.reshape(-1, 1)])
        new_train_prediction_out_df = prediction_outtable_df(train_DATA_new.astype(np.float64), new_train_pred_ratings, new_train_pred_ratings_std)
        # print(new_train_prediction_out_df.describe())
        # print(new_train_prediction_out_df.tail(10))
        new_train_RMSE = RMSE_val(new_train_pred_ratings, y_new_training_scaled)
        # print('*************RMSE of new train: ', new_train_RMSE) 
        # 
        ########################################
        ### output new test predictions        
        # print('New Test Prediction: ................')
        # X_test_update_forcal = X_test_update
        # X_test_update_forcal[:, 3] = X_test_update_forcal[:, 3] * 100.0
        new_test_prediction_out_df = prediction_outtable_df(X_test_update.astype(np.float32), new_test_pred_ratings, new_test_pred_ratings_std)
        # new_test_prediction_out_df2 = new_test_prediction_out_df.dropna().reset_index(drop=True)
        # print(new_test_prediction_out_df.describe())
        # print('Split Test statistics ###########################################')
        # print(new_test_prediction_out_df.tail(10))

        new_test_RMSE = RMSE_val(new_test_pred_ratings, X_test_update[:, 3] * 100)
        # print('*************RMSE of new test: ', new_test_RMSE)    
    

        return new_train_pred_ratings, new_train_pred_ratings_std, new_test_pred_ratings, new_test_pred_ratings_std, new_used_GPU0_lst, new_used_GPU1_lst, new_used_GPU2_lst, new_train_prediction_out_df, new_train_RMSE, new_test_prediction_out_df, new_test_RMSE

# %%
def plot_RMSE(target_RMSE_lst, save_path, track):
    # summarize history for RMSE
    plt.plot(range(1, len(target_RMSE_lst)+1), target_RMSE_lst)
    plt.title('Uncertainty RMSE Patterns during Active learning')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend([track], loc='upper right')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()
    plt.close()
# %%



# %%


# %%
if __name__ == '__main__': 

    # rng_key = random.PRNGKey(101)
    ######### Check GPU usage ##############
    GPU_mem_state=None
    try:
        nvmlInit()
        GPUhandle_gpu0 = nvmlDeviceGetHandleByIndex(0)
        GPUhandle_gpu1 = nvmlDeviceGetHandleByIndex(1)
        GPUhandle_gpu2 = nvmlDeviceGetHandleByIndex(2)
        numpyro.set_platform("gpu")
        
        def GPU_mem_state():
            info_gpu0 = nvmlDeviceGetMemoryInfo(GPUhandle_gpu0)
            # print('GPU 0: ', info_gpu0)
            info_gpu1 = nvmlDeviceGetMemoryInfo(GPUhandle_gpu1)
            # print('GPU 1: ', info_gpu1)
            info_gpu2 = nvmlDeviceGetMemoryInfo(GPUhandle_gpu2)
            # print('GPU 2: ', info_gpu2)
            
            info_gpu0_mb = np.round(info_gpu0.used/1000000,4)
            info_gpu1_mb = np.round(info_gpu1.used/1000000,4)
            info_gpu2_mb = np.round(info_gpu2.used/1000000,4)
             
            print("Used GPU Memory MB og GPU_0: {:,}; Used GPU Memory MB og GPU_1: {:,}; Used GPU Memory MB og GPU_2: {:,};".format(info_gpu0_mb, info_gpu1_mb, info_gpu2_mb))

            return info_gpu0_mb, info_gpu1_mb, info_gpu2_mb
            # return "Used GPU Memory MB {:,}".format(np.round(info.used/1000000,6))
    except:
        print ("Cant initialise GPU, Using CPU")
    
    # ######### Check GPU usage ##############

    ### read in training and test data as well as side features
    main_tensor_dir = "/home/faceal/data.csv"
    visual_face_dir = "/home/faceal/face_vectors.npy"
    language_trait_dir = "/home/faceal/language_vectors.npy"
    # subset_size = 1134836
    DATA_train_init, DATA_test_init, DATA_all, visual_face_vectors, language_trait_vectors = input_data_process(main_tensor_dir, 
                                                                                                                visual_face_dir, language_trait_dir, subset_size = 1134836, train_size = 8)

    all_indexes, logit_all_rates = DATA_all[:, :-2].astype(np.int32), logit(DATA_all[:, 3])
    print('all data indexes: -------')
    # print(all_indexes.dtype)
    print(all_indexes.shape)

    all_participants = len(np.unique(DATA_all[:, 0]))  # number of participants (was K)
    print('no. of all participants: ', str(all_participants))
    all_stimulus = len(np.unique(DATA_all[:, 1]))  # number of stimuli (was N)
    print('no. of all stimulus: ', str(all_stimulus))
    all_traits = len(np.unique(DATA_all[:, 2]))  # number of traits (was T)
    print('no. of all traits: ', str(all_traits))
    all_responses = len(DATA_all)  # number of responses (was R)
    print('no. of all responses: ', str(all_responses))



    ### Process training data
    DATA_train = DATA_train_init
    train_indexes, logit_train_rates = DATA_train[:, :-2].astype(np.int32), logit(DATA_train[:, 3])
    print('training data indexes: -------')
    # print(train_indexes.dtype)
    print(train_indexes.shape)
    print('logit_train_rates: -------')
    # print(DATA_train[:, 3] )
    # print(expit(logit_train_rates))
    print('logit train rates min: ', np.min(logit_train_rates))
    print('logit train rates max: ', np.max(logit_train_rates))

    #################################################
    #################################################

    #### Process test data
    ### import test data
    DATA_test = DATA_test_init
    # print(DATA_test.shape)
    test_indexes, logit_test_rates = DATA_test[:, :-2].astype(np.int32), logit(DATA_test[:, 3])
    print('test indexes: -------')
    # print(test_indexes.dtype)
    print(test_indexes.shape)
    print('logit_test_rates: -------')
    # print(DATA_test[:, 3] )
    # print(expit(logit_test_rates))
    print('logit test rates min: ', np.min(logit_test_rates))
    print('logit test rates max: ', np.max(logit_test_rates))

    #################################################
    #################################################

    latent_dim = 10  # The dimensionality of the latent space (was DIMENSIONS)
    ### if choose mcmc simulation
    # mcmc_posterior_samples, 
    train_pred_ratings, train_pred_ratings_std, test_pred_ratings, test_pred_ratings_std, used_GPU0_lst, used_GPU1_lst,used_GPU2_lst = bayesian_mcmc_run_predict(model_input = model,
                                                                                                                                        features = train_indexes,
                                                                                                                                        participants = all_participants,
                                                                                                                                        stimulus= all_stimulus,
                                                                                                                                        traits= all_traits,
                                                                                                                                        latent_dimensions=latent_dim,
                                                                                                                                        rates= logit_train_rates,
                                                                                                                                        features_test = test_indexes, 
                                                                                                                                        participants_test = all_participants,
                                                                                                                                        stimulus_test = all_stimulus,
                                                                                                                                        traits_test = all_traits,
                                                                                                                                        latent_dimensions_test =latent_dim
                                                                                                                                        )

    # print(predictions_jit.shape)
    # print(type(predictions_jit))
    # print(type(mcmc_posterior_samples))

    # used_gpu0_train_overseq_dir = './output_result/testcase_used_GPU0_training.jpg'          
    # plot_gpu_usage(used_GPU0_lst, save_path = used_gpu0_train_overseq_dir)
    
    # used_gpu1_train_overseq_dir = './output_result/testcase_used_GPU1_training.jpg'          
    # plot_gpu_usage(used_GPU1_lst, save_path = used_gpu1_train_overseq_dir)

    # used_gpu2_train_overseq_dir = './output_result/testcase_used_GPU1_training.jpg'          
    # plot_gpu_usage(used_GPU2_lst, save_path = used_gpu2_train_overseq_dir)
    #######################################
    #######################################
    ### output training predictions
    # print('Initial Training Prediction: ................')
    train_prediction_out_df = prediction_outtable_df(DATA_train, train_pred_ratings, train_pred_ratings_std)
    # print(train_prediction_out_df.describe())
    # print(train_prediction_out_df.tail(10))

    # print('######################################################')
    # print('train_pred_ratings: ', train_pred_ratings)
    # print('train_orig_ratings: ', DATA_train[:, 3] * 100.0)
    train_RMSE = RMSE_val(train_pred_ratings, DATA_train[:, 3] * 100.0)
    print('*************RMSE of initial train: ', train_RMSE)

    #######################################
    #######################################
    ### output test predictions
    # print('Initial Test Prediction: ................')
    test_prediction_out_df = prediction_outtable_df(DATA_test, test_pred_ratings, test_pred_ratings_std)
    # print(test_prediction_out_df.describe())
    # print(test_prediction_out_df.tail(10))

    test_RMSE = RMSE_val(test_pred_ratings, DATA_test[:, 3] * 100.0)
    print('*************RMSE of initial test: ', test_RMSE)

    print('after 1 epoch GPU cost:---------------------------------------------------------')
    info_gpu0_mb_take3, info_gpu1_mb_take3, info_gpu2_mb_take3 = GPU_mem_state()

    print('----------------------------------------start active learning process  -----------------------------------------------------------')
    ### Initalize the new training data (old + newly queried) with old training data
    n_queries = 2500

    ### For greedy 
    org_X_greedy = DATA_train[:, 0:3]
    org_y_greedy = DATA_train[:, 3]
    Data_test_select_greedy  = DATA_test
    potential_X_select = DATA_test[:, 0:3]
    potential_y_select = DATA_test[:, 3]
    
    # mcmc_posterior_samples_select_maxstd = mcmc_posterior_samples
    test_pred_ratings_select_greedy = test_pred_ratings 
    # print('******************************initial test_pred_ratings_select_maxstd: ', test_pred_ratings)
    test_pred_ratings_std_select_greedy = test_pred_ratings_std

    train_RMSE_lst_greedy = []
    test_RMSE_lst_greedy = []
    train_RMSE_lst_greedy.append(train_RMSE)
    test_RMSE_lst_greedy.append(test_RMSE)


    GPU0_avaiable_mb_lst_greedy = used_GPU0_lst
    GPU1_avaiable_mb_lst_greedy = used_GPU1_lst
    GPU2_avaiable_mb_lst_greedy = used_GPU2_lst

    est_running_time = 0.0
    
    for q in range(n_queries):
        start_time_per_epochs = time.time()

        print('****************************!!!!!! --Epoch: ' + str(q) + '-- !!!!!!!!**************************************')
        ###########################################################################################################
        ###########################################################################################################
        #####################################sample selection ######################################################
        ########### greedy sampling
        ### when run batch
        print('GPU cost within epoch:--- greedy sampling ----------------------------')
        info_gpu0_mb_take0, info_gpu1_mb_take0, info_gpu2_mb_take0 = GPU_mem_state()
        print('--------------------------------------------------------------------------')
        kgreedy = kg.kCenterGreedy(org_X_greedy, org_y_greedy, potential_X_select, potential_y_select, seed = 42)
        queried_idx_greedy, queried_data_test_greedy, X_test_new_greedy = kgreedy_aggree_TopN(Data_test_select_greedy, 
                                                                            new_test_ratings = test_pred_ratings_select_greedy,
                                                                            new_test_ratings_std = test_pred_ratings_std_select_greedy, 
                                                                            select_batch_func = kgreedy.select_batch_, 
                                                                            already_selected = kgreedy.already_selected,
                                                                            top_N = 8
                                                                            )
        
        Data_test_select_greedy = X_test_new_greedy
        potential_X_select, potential_y_select = X_test_new_greedy[:, 0:3], X_test_new_greedy[:, 3]
        test_temp_dir = '/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/v20_greedy_test_temp_rd11_batch8.npy'
        np.save(test_temp_dir , X_test_new_greedy)
        # print('queried outputs-----------------:')
        # print('index: ', queried_idx_maxstd)
        # print('returned labelled data: ', queried_data_test_greedy)
        # print('querying process  ended-----------------:')
        # print('~~~~~~~~~Original training y:', org_y_greedy)
#         ### when run batch
#         X_new_training_greedy, y_new_training_greedy = _add_training_data(org_X_greedy, org_y_greedy, X = queried_data_test_greedy[:, 0:3].reshape(-1,3), y= queried_data_test_maxstd[:, 3].reshape(-1))  
        
#         org_X_greedy, org_y_greedy = X_new_training_greedy, y_new_training_greedy

#         train_X_temp_dir = '/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/v20_greedy_train_X_temp_rd1_batch8.npy'
#         train_y_temp_dir = '/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/v20_greedy_train_y_temp_rd11_batch8.npy'
#         np.save(train_X_temp_dir , X_new_training_greedy)
#         np.save(train_y_temp_dir , y_new_training_greedy)
# #         # print('~~~~~~~~~Partly new training y:', y_new_training[-11+q:])
#         # print('*******************test here !!!!!')
#         # print('~~~~~~~~~New test set:', Data_test_select_greedy)
#         # print('~~~~~~~~~shape of New test set:', Data_test_select_greedy.shape)
#         # print('~~~~~~~~~New training y:', y_new_training_greedy)
#         # print('~~~~~~~~~Check new training X shape after queried a set of points:', X_new_training_greedy.shape)
#         # print('~~~~~~~~~Check new training y shape after queried a set of points:', y_new_training_greedy.shape)

#         ###########################################################################################################
#         ###########################################################################################################
#         #####################################run experiemnts ######################################################
#         ########### greedy sampling
#         ### fit the new training data to bayesian model with fixed prior params to get updated posteriors parms
#         print('New Training Prediction for max greedy ................')
#         new_train_pred_ratings_greedy, new_train_pred_ratings_std_greedy, new_test_pred_ratings_greedy, new_test_pred_ratings_std_greedy, new_used_GPU0_lst_greedy, new_used_GPU1_lst_greedy, new_used_GPU2_lst_greedy, new_train_prediction_out_df_greedy, new_train_RMSE_greedy, new_test_prediction_out_df_greedy, new_test_RMSE_greedy = fit_to_mcmc(X_new_training_greedy, y_new_training_greedy, Data_test_select_greedy)
#         GPU0_avaiable_mb_lst_greedy += new_used_GPU0_lst_greedy
#         GPU1_avaiable_mb_lst_greedy += new_used_GPU1_lst_greedy
#         GPU2_avaiable_mb_lst_greedy += new_used_GPU2_lst_greedy

#         print('************* Statistics of new train: ')
#         # print(new_train_prediction_out_df_maxstd.describe())
#         print('************* RMSE of new train: ', new_train_RMSE_greedy) 
#         print('************* Statistics of new test: ')
#         print('new_test_pred_ratings_maxstd:', new_test_pred_ratings_greedy)
#         # print(new_test_prediction_out_df_maxstd.describe())
#         print('************* RMSE of new test: ', new_test_RMSE_greedy)

#         train_RMSE_lst_greedy.append(new_train_RMSE_greedy)
#         test_RMSE_lst_greedy.append(new_test_RMSE_greedy)

#         RMSE_output_dict_greedy = dict()
#         RMSE_output_dict_greedy['train_RMSE_greedy'] = train_RMSE_lst_greedy
#         RMSE_output_dict_greedy['test_RMSE_greedy'] = test_RMSE_lst_greedy

#         ### when run batch
#         with open('/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/v20_greedy_RMSE_outputs_train_test_batch8_rd1_v2.pickle', 'wb') as dt:
#             pickle.dump(RMSE_output_dict_greedy, dt, protocol=pickle.HIGHEST_PROTOCOL)

#         ### plot RMSE for training vs test
#         train_RMSE_plot_dir = "/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/train_AL_RMSE_plot_v20_greedy_batch8_rd1_v2.jpg"
#         plot_RMSE(target_RMSE_lst = train_RMSE_lst_greedy, save_path = train_RMSE_plot_dir, track ='Train')

#         test_RMSE_plot_dir = "/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/test_AL_RMSE_plot_v20_greedy_batch8_rd1_v2.jpg"
#         plot_RMSE(target_RMSE_lst = test_RMSE_lst_greedy, save_path = test_RMSE_plot_dir, track ='Test')

       
#         ### plot used GPU for GPU 0 and 2
#         used_gpu0_train_overseq_dir = '/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/testcase_used_GPU0train_AL_RMSE_plot_v20_greedy_batch8_rd1_v2.jpg'          
#         plot_gpu_usage(GPU0_avaiable_mb_lst_greedy, save_path = used_gpu0_train_overseq_dir)
       

#         used_gpu1_train_overseq_dir = '/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/testcase_used_GPU1train_AL_RMSE_plot_v20_greedy_batch8_rd1_v2.jpg'          
#         plot_gpu_usage(GPU1_avaiable_mb_lst_greedy, save_path = used_gpu1_train_overseq_dir)

#         used_gpu2_train_overseq_dir = '/home/faceal/v21_al_run_3rds_greedy_batch8/output_results_summary/testcase_used_GPU2train_AL_RMSE_plot_v20_greedy_batch8_rd1_v2.jpg'          
#         plot_gpu_usage(GPU2_avaiable_mb_lst_greedy, save_path = used_gpu2_train_overseq_dir)

#         ###########################################################################################################
#         ###########################################################################################################
#         #####################################updating the parameters for looping ########################################


#         ### update the posterior and newly predicted ratings and stds after fitting in the new training data at each query step
        # test_pred_ratings_select_greedy = new_test_pred_ratings_greedy
        # test_pred_ratings_std_select_greedy = new_test_pred_ratings_std_greedy

        print('*******************************************************************************')
        print("******------------- %s seconds of completing one epoch time -----------------****" % (time.time() - start_time_per_epochs))
        print('*******************************************************************************')
        est_running_time += (time.time() - start_time_per_epochs)
        print("******------------- %s seconds of total running time by far -----------------****" % est_running_time)
        print('*******************************************************************************')
# # %%
