#!/usr/bin/env python
# coding: utf-8


import numpy as np
import copy


def convert_coordinates_by_level(level0,side_tiles, actual_level, actual_x,actual_y):

	'''
	Input: original level, number of tiles in this level, current level and current positions
	Output: position in the base level
	'''
	
	actual_num_sides = (2**actual_level)
	num_units = side_tiles//actual_num_sides
	base_x = actual_x*num_units
	base_y = actual_y*num_units

	return base_x, base_y, num_units

def convert_coordinates_by_node(mask_size, node,tree):

	'''
	Input: mask size = image size, number of the node, tree
	Output: generated mask
	'''

	mask_aux = np.full(mask_size, False)

	leaves = [i for i in tree.leaves() if node in tree.ancestor(i)]

	for l in leaves:
	
		mask_aux[l//mask_size[1], l%mask_size[1]] = True

	return mask_aux


def sum_values_matrix(matrix, base_x, base_y, num_units,value):

	'''
	Input: matrix, base coordinates, num of units for each coordinate in this level, value to sum
	Output: matrix with value summed up in the correct position tiles
	'''

	matrix[base_x:(base_x+num_units), base_y:(base_y+num_units)] += value 

	return matrix


def norm_matrix_max(matrix):

	'''
	Input: matrix to normalize
	Output: normalized by minmax
	'''

	matrix_max = np.max(matrix)
	matrix_min = np.min(matrix)

	return (matrix-matrix_min)/ (matrix_max-matrix_min+0.0001)


def chose_next(actual_level,matrix_local_values,threshold):

	'''
	Input: current level, matrix of values and threshold in percentage
	Output: coordinates of most important tiles according to threshold, value of threshold calculated based on the percentage threshold and the matrix values
	'''
	#try:
	actual_num_sides = (2**actual_level)
	threshold_value = (np.max(matrix_local_values) - np.min(matrix_local_values))*threshold + np.min(matrix_local_values)
	#print(matrix_local_values,matrix_local_values.shape)
	coord = np.array([[x,y] for x in range(actual_num_sides) for y in range(actual_num_sides) if (matrix_local_values[x,y] >= threshold_value)])
	#except:
	#	coord = []
	#	threshold_value = 0

	return coord, threshold_value

def chose_next_mask(coord_current,matrix_local_values,threshold):

	'''
	Input: current coordinates, matrix of values and threshold in percentage
	Output: coordinates of most important tiles according to threshold, value of threshold calculated based on the percentage threshold and the matrix values
	'''
	#try:
	threshold_value = (np.max(matrix_local_values) - np.min(matrix_local_values))*threshold + np.min(matrix_local_values)
	new_coord = []

	keep = [list(x) for i,x in enumerate(coord_current) if (matrix_local_values[coord_current[i]].mean() >= threshold_value)]
	#print(keep)
	for seg in keep:

		for i in range(len(matrix_local_values)):
			
			if i not in seg:
				#print(seg)
				aux = copy.copy(seg)
				aux.append(i)
				#print(aux)
				new_coord.append(aux)
	#print(new_coord)

	return new_coord, threshold_value


	
def chose_next_max(level0,side_tiles,actual_level,matrix_local_values):

	'''
	Input: base level, number of tiles per side, current level,matrix of values
	Output: coordinates of max importance
	'''


	coord = np.argmax(matrix_local_values)
	coord = [coord//len(matrix_local_values), coord%len(matrix_local_values)]
	x,y,units = convert_coordinates_by_level(level0,side_tiles, actual_level, coord[0],coord[1])
	return [coord],[x,y]


