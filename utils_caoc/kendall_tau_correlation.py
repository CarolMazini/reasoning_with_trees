#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
from scipy.stats import kendalltau


def count_changes(rank_a, rank_b,img_id,img_id2 = None):
	'''
	Input: two rankings (same original samples' positions with the order position), the positions to be considered in both rankings
		if img_id2 is None, we considered the same idx in the ranking
	Output: absolute difference between the two positions, signal indication if the position goes up or down in the ranking 
	'''
	
	if img_id2 == None: img_id2 = img_id
	count_changes = rank_b[img_id] - rank_a[img_id2]
	#print(count_changes)
	
	signal = 1 if count_changes >= 0 else -1 

	return np.abs(count_changes), signal
	

def kendall_tau(rank_a, rank_b):
	'''
	Input: two rankings (same original samples' positions with the order position)
	Output: kendall_tau correlation
	'''

	return kendalltau(rank_a, rank_b).correlation



def calculate_units_correlation(classes,unit_a, unit_b):
	'''
	Input: num of classes to be correlated, two ordem vectors corresponding to units a and b
	Output: a vector of correlations between classes for units a and b, there is one correlation value per class
	'''
	
	corr_classes = np.zeros(classes)

	for c in range(classes):
		print('class', c)

		class_bool = np.array(unit_a[0,:] == np.full(unit_a[0,:].shape, c), dtype=bool)

		print('Num classe:', class_bool.sum())

		corr_classes[c] = np.abs(kendall_tau(unit_a[1,class_bool], unit_b[1,class_bool]))

		
	print(corr_classes)
	return corr_classes


def ranking_by_correlation(data,order_corr,type_corr='separated'):
	'''
	Input: dataframe with correlation by class for all units with respect to one another single one, order for classes to be considered, if the type of correlation will be separated or summed
	Output: same initial dataframe with an extra column containing the sum correlation (if typ_corr == sum) and ordered by correlation
	'''
	
	if type_corr == 'separated':
		order_by = order_corr

	elif type_corr == 'sum':

		data['sum_corr'] = np.zeros(len(data))

		for classe in order_corr:
			data['sum_corr'] += data[classe]
		order_by = ['sum_corr']
		

	
	data = data.sort_values(by=order_by, ascending=False).reset_index()

	return data



