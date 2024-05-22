#!/usr/bin/env python
# coding: utf-8


"""
Credit to https://github.com/CarolMazini/unsupervised-IVC

Caroline Mazini Rodrigues, Nicolas Boutry, Laurent Najman, Unsupervised discovery of interpretable visual concepts,
Information Sciences,Volume 661,2024,https://doi.org/10.1016/j.ins.2024.120159.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import sys
import os
from multiprocessing import Process, Pool
np.set_printoptions(threshold=sys.maxsize)
import utils_caoc.kendall_tau_correlation as kendall_tau_correlation
import utils_caoc.order_functions as order_functions
import causal_viz.images_by_patches as images_by_patches
import concurrent.futures
import torchvision.transforms as transforms
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras import backend as keras


BATCH_SIZE = 16
WORKERS = 4
NUM_THREADS = 4


def get_preprocess_transform(size_img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((size_img, size_img)),
	normalize
    ])    

    return transf,normalize



def multi_scale_viz_simple(net, base_images,images_file,rank_original,probs_original, num_images, labels, classes, level0, level_final, idx_img, size_tile,thr,complete_space = False, device = 'cpu',sample=None):


	
	#trans = transforms.Compose([transforms.ToTensor()]) 
	transf,normalize = get_preprocess_transform(level0)


	if not base_images==None:
		image_path = images_file.iloc[idx_img]['path']
		sample =Image.open(base_images+image_path, mode='r').convert("RGB")
	else:

		sample=sample
		

	sample = sample.resize((level0,level0))
		
	#num_tiles_side = (self.size_img// self.size_tile)
	num_final_side_tiles = level0 // size_tile                             #num of smallest size tile per image side                          
	number_tiles_top = num_final_side_tiles * num_final_side_tiles         #total number of smallest size tiles 
	leveln = int(np.sqrt(level0 // level_final))                           #last level in the hierarchy

	matrix_weights = np.zeros((num_final_side_tiles, num_final_side_tiles))

	#--function to split tile in coord vector into four other tiles and 
	#--calculate importance
	def process_coordinate(coord):
		#print('T')

		matrix_weights_aux = np.zeros((num_final_side_tiles, num_final_side_tiles))
		matrix_local_values = np.zeros((num_tile_level, num_tile_level))
		


	
		for i in range(coord[0] * 2, coord[0] * 2 + 2):
			for j in range(coord[1] * 2, coord[1] * 2 + 2):

				base_x, base_y, num_units = images_by_patches.convert_coordinates_by_level(level0, num_final_side_tiles, level, i, j)	#relative coordinates to global coordinates			
				sample_copy = sample.copy()
				
				result = Image.new(sample_copy.mode, (num_units*size_tile, num_units*size_tile), (0, 0, 0))
				sample_copy.paste(result, (base_x*size_tile, base_y*size_tile))

				
				sample_copy = transf(sample_copy)

				#--changing activation for one tile occluded image
				
				probs_modified = probs_original.copy()
				prob_idx = net(sample_copy.unsqueeze(0).to(device))
				probs_modified[idx_img,:] = prob_idx[0,:].detach().cpu().numpy()  
				#--


				#--CaOC
				feat_map_probs_occlusion = order_functions.create_order(probs_modified, labels, classes,space=complete_space,all_samples = complete_space)      #create new order
				new_corr, signal = kendall_tau_correlation.count_changes(rank_original[1, :], feat_map_probs_occlusion[1, :], idx_img)
				#--


				matrix_weights_aux = images_by_patches.sum_values_matrix(matrix_weights_aux, base_x, base_y, num_units, new_corr) #update importances in the aux matrix of tiles
				matrix_local_values[i, j] += new_corr #accumulate sum
	

		return matrix_weights_aux, matrix_local_values

	#--
	#--



	selected_patches = np.array([[0, 0]])   #we start with the complete image as a tile

	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:     #parallelizing importance calculus
		for level in range(1, leveln + 1):
			
			num_tile_level = (2 ** level)
			matrix_local_values = np.zeros((num_tile_level, num_tile_level))

			coord_data = []
			for coord in selected_patches:
				coord_data.append(coord)

			if len(coord_data)>50:
				break

			results = executor.map(process_coordinate, coord_data)
			matrix_weights_aux2 = np.zeros((num_final_side_tiles, num_final_side_tiles))
			
			for result in results:                                             #combining results to obtain final hierarchical importance matrix
				#print(result)
				
				matrix_weights_aux, matrix_local_values_aux = result
				matrix_local_values+= matrix_local_values_aux
				matrix_weights_aux2 += matrix_weights_aux

			matrix_weights += images_by_patches.norm_matrix_max(matrix_weights_aux2)   #norm by max importance value of the level
			selected_patches, threshold_value = images_by_patches.chose_next(level, matrix_local_values, thr)   #chose patches for next level according to threshold

	matrix_weights = images_by_patches.norm_matrix_max(matrix_weights)

	
	return matrix_weights

##############################################
#                Ms-IV                       #
##############################################





	
