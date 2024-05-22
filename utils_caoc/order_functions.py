#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
import os


def all_images_ranking(activations_sum, size_tile, num_feat_maps):


	'''
	Input: matrix of activations for all tile images (rows) for all feature maps (columns), size of the tiles and list of feature map indexes to be considered 
	Output: matrix of tile aggregated rankings (biggest activations first) for all images
	'''


	num_tiles_side = (224// size_tile)

	final_image_ranking = []

	for i in tqdm(range(0,len(activations_sum), num_tiles_side*num_tiles_side)):

		activations_one_image = [] #for all feature maps


		for feat_map in num_feat_maps:

			
			activations_one_image.append(activations_sum[i:(i+ num_tiles_side*num_tiles_side),feat_map])

		final_image_ranking.append(ranking_agg_by_borda(activations_one_image))


	return np.array(final_image_ranking)

def ranking_agg_by_borda(vec_activations):

	'''
	Input: list of activations for one image and all feature maps
	Output: aggregated ranking (using simple aggregation method borda) of the image tiles according to biggest activations of all features maps
	'''

	rank = np.zeros(len(vec_activations[0]))
	for i in range(len(vec_activations)):


		order = np.argsort(-vec_activations[i])

		rank[order] += np.arange(len(order)).reshape(-1,)

	return np.argsort(rank)


def create_order(complete_activations,label, classes, space=False,all_samples = False):
	'''
	Input: activations, corresponding labels from each sample in the activations order and number of classes, or vector of classes analyzed
	Output: matrix with first column filled with each sample label, second column with the corresponding order in the label's order
	'''


	
	orders = {}
	

	rank_save = np.full((2,complete_activations.shape[0]), np.inf)

	if space:
		distances = np.linalg.norm(complete_activations,ord=2, axis = 1)

	if type(classes) == int:
		classes = np.arange(classes)

	labels_bool = np.empty((len(classes),complete_activations.shape[0]), dtype=bool)
		

	for j,c in enumerate(classes):

		if all_samples:
			labels_bool[j,:] = np.full(complete_activations.shape[0], True)

		else:
			labels_bool[j,:] = np.array(label == np.full(label.shape, c), dtype=bool)

		aux = np.where(labels_bool[j,:])[0]

		if space:
			
			orders[c] = np.array([int(i) for i in np.argsort(distances) if i in aux])

		else:

			orders[c] = np.array([int(i) for i in np.argsort(-complete_activations[:,c]) if i in aux])
		

		
		rank_save[0,orders[c]] = np.full((orders[c].shape), c)
		rank_save[1,orders[c]] = np.arange(len(orders[c])).reshape(-1,)
	
	return np.array(rank_save)






