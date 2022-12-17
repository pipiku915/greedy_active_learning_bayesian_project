# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""
#### Note: Merge 2 files: sampling_def and kcenter greedy together
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as jnp
import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
import abc
# from sampling_methods.sampling_def import SamplingMethod

### from file 1: sampling def

"""Abstract class for sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.

"""

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, org_X, org_y, potential_X, potential_y, seed):     #, **kwargs):
    self.org_X = org_X.astype(np.int32)
    self.org_y = org_y
    self.potential_X = potential_X.astype(np.int32)
    self.potential_y = potential_y
    self.seed = seed

  def feature_extract_X(self):
    ### import language and image feature data
    print('Loading side feature datasets ......')
    visual_face_dir = "/home/faceal/face_vectors.npy"
    language_trait_dir = "/home/faceal/language_vectors.npy"
    visual_face_vectors = np.load(visual_face_dir,allow_pickle=True)
    language_trait_vectors = np.load(language_trait_dir, allow_pickle=True)
    visual_face_vectors = jnp.array(visual_face_vectors)
    language_trait_vectors = jnp.array(language_trait_vectors)

    ### Create feature matrix for orginal training data 
    ### and potential selected data pool (test data)
    org_visual_features = visual_face_vectors[self.org_X[:, 1] - 1]
    org_language_features = language_trait_vectors[self.org_X[:, 2] - 1]
    # print('Visual feature shape of original training set: ', org_visual_features.shape)
    # print('Language feature shape of original training set: ', org_language_features.shape)

    potential_visual_features = visual_face_vectors[self.potential_X[:, 1] - 1]
    potential_language_features = language_trait_vectors[self.potential_X[:, 2] - 1]
    # print('Visual feature shape of potential pool: ', potential_visual_features.shape)
    # print('Language feature shape of potential pool: ', potential_language_features.shape)

    org_all_features = jnp.concatenate((org_visual_features, org_language_features), axis = 1)
    potential_all_features = jnp.concatenate((potential_visual_features, potential_language_features), axis = 1)
    # shape = self.X.shape
    # flat_X = self.X
    # if len(shape) > 2:
    #   flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return org_all_features, potential_all_features, org_visual_features, org_language_features,  potential_visual_features, potential_language_features 


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None



#####################################################################

### from file 2:kcenter greedy

class kCenterGreedy(SamplingMethod):

  def __init__(self, org_X, org_y, potential_X, potential_y, seed, metric='euclidean'):
    super().__init__(org_X, org_y, potential_X, potential_y, seed)
    self.org_X = org_X.astype(np.int32)
    self.org_y = org_y
    self.potential_X = potential_X.astype(np.int32)
    self.potential_y = potential_y
    # self.flat_X = self.flatten_X()
    self.org_all_features, self.potential_all_features,_, _, _, _ = self.feature_extract_X()
    self.name = 'kcenter'
    self.org_features, self.potential_features = self.org_all_features, self.potential_all_features
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.potential_X.shape[0] ### still ???
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      # print('cluster centers: ', cluster_centers)

      x = self.org_features[np.array(cluster_centers)]
      dist = pairwise_distances(self.potential_features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)


  def select_batch_(self, already_selected, N):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    try:

      # Assumes that the transform function takes in original data and not
      # flattened data.
      print('Getting transformed features...')
      self.potential_features = self.potential_features
      # print('potential_features: ', self.potential_features.shape)
      # print('Already selected: ', already_selected)

      print('Calculating distances for original + newly selected samples...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)

    except:

      print('Using only newly selected as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []

    for _ in range(N):
      # if self.already_selected is None:
      if self.already_selected == []:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
        # print('selected index: ', ind)

      else:
        print('Min distance: ', self.min_distances)
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.

      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
    # print('*********************************************')
    # print('A comparison: ')
    # print(self.already_selected)
    # print('--------------------------------')
    # print(already_selected)
    # print('*********************************************')
    self.already_selected = already_selected + new_batch

    return new_batch

