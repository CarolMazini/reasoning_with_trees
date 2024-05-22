#!/usr/bin/env python
# coding: utf-8

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
import utils_model.test_individual_feat_map as test_individual_feat_map
import concurrent.futures
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import higra as hg
from skimage.util import img_as_float
from skimage.filters import sobel, laplace
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())
import cv2
from PIL import Image
from captum.attr import IntegratedGradients,GuidedBackprop, InputXGradient, Saliency

detector = cv2.ximgproc.createStructuredEdgeDetection(get_sed_model_file())
BATCH_SIZE = 128
WORKERS = 1
NUM_THREADS = 4
from skimage.transform import resize




def attribute_image_features(net,algorithm,label,input, **kwargs):
    """
    Function to obtain the attribution map from a xAI pixe-wise techniquel,
    a model and an input image
    """
    net.zero_grad()
    input.requires_grad=True
    try:
	    tensor_attributions = algorithm.attribute(input,
		                                      target=label,
		                                      **kwargs
		                                     )
    except:
            tensor_attributions = algorithm.attribute(input,
		                                      target=label)

    input.requires_grad=False
    return tensor_attributions




def get_preprocess_transform(size_img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((size_img, size_img)),
	normalize
    ])    

    return transf



def mask_nodes(t1b,altitudesb,image,areamin=500):
	"""
	Function to generate the occlusion dataset based on the segmentation regions
	"""
	alt = np.unique(altitudesb)
	dataset_occ = []
	parts_visited = []
	#use sub_tree with the node_map
	size = image.shape
	for n in t1b.leaves_to_root_iterator(include_leaves = False,include_root = True):
		st,node_map = t1b.sub_tree(n)
		image_copy = image.copy().reshape(-1,3)
		node_map = node_map[node_map<image_copy.shape[0]]
		image_copy[node_map,:] = 0
		dataset_occ.append(image_copy.reshape(size))

	return dataset_occ


class dataset_masks(Dataset):
	"""
	Class of the occlusion dataset
	"""
	def __init__(self,images, transform=None, size_img = 224):

		self.images = images
		self.transform = transform
		self.size_img = size_img

	def __getitem__(self, idx):
		sample = self.images[idx]

		if self.transform:
			sample = self.transform(sample)
		
		#print(sample.shape)
		return sample.float(), 0, 0

	def __len__(self):
		return len(self.images)


def buildTree(graph, edge_weights=None, areamin=500, tree="BPT",img_size = 224,method_explain=None,attribute = None,image=None, net=None):
    """
    Function to build the hierarchical segmentation tree T and filter out small regions

    Credit to https://github.com/higra/Higra-Notebooks/
	Yongchao Xu, Edwin Carlinet, Thierry Géraud, Laurent Najman. Hierarchical Segmentation Using Tree-Based Shape Spaces.
	 IEEE Transactions on Pattern Analysis and Machine Intelligence, Institute of Electrical and Electronics Engineers,
	 2017, 39 (3), pp.457-469. ⟨10.1109/TPAMI.2016.2554550⟩. ⟨hal-01301966⟩
    """
    if tree == 'watershed':
        if 'explain' in attribute:
            preprocess_transform = get_preprocess_transform(size_image)
            label = net(preprocess_transform(image).unsqueeze(0)).detach().cpu().numpy()
            print(label,label.argmax(axis=1))
            attr_explain, delta = attribute_image_features(method_explain,int(label.argmax(axis=1)[0]), preprocess_transform(image).unsqueeze(0).to(device), baselines=preprocess_transform(np.zeros((img_size,img_size,3),dtype=np.float32)).unsqueeze(0).to(device), return_convergence_delta=True)
            attr_explain = np.transpose(attr_explain.squeeze().cpu().detach().numpy().copy(), (1, 2, 0)).mean(axis=2)
            attr_explain[attr_explain<0] = 0
            t1a, altitudesa = hg.watershed_hierarchy_by_attribute(graph, edge_weights, lambda tree, altitudes:hg.attribute_area(tree,vertex_area=attr_explain.mean()+attr_explain))
        else:
            t1a, altitudesa = eval(attribute)(graph, edge_weights)
            
        t1b, altitudesb = hg.filter_small_nodes_from_tree(t1a, altitudesa, areamin)
        t1, altitudes = hg.canonize_hierarchy(t1b, altitudesb)

        return t1, altitudes

    elif tree == "BPT":
        # --- Build the BPT
        t1a, altitudesa = hg.bpt_canonical(graph, edge_weights)
        # and remove small regions from it
        t1b, altitudesb = hg.filter_small_nodes_from_tree(t1a, altitudesa, areamin)
        t1, altitudes = hg.canonize_hierarchy(t1b, altitudesb)
        return t1, altitudes
    else:
        if tree == "ToS":
            t1a, altitudesa = hg.component_tree_tree_of_shapes_image2d(image)
        elif tree == 'maxTree':
            t1a, altitudesa = hg.component_tree_max_tree(graph, image)
        elif tree == "minTree":
            t1a, altitudesa = hg.component_tree_min_tree(graph, image)
        else:
            raise ValueError("Error. Possible choices are BPT, maxTree, minTree or ToS")
        area = hg.attribute_area(t1a)
        condRemoval = area < areamin

        t1, node_map = hg.simplify_tree(t1a, condRemoval)
        a1 = altitudesa[node_map]
        return t1, a1



def occ_attribute(net,tree_ini,attribute,image,idx_img,size_image, complete_space = False,areamin= 500,device= 'cpu',gradient_method = None):
	"""
	Occlusion attribute computation for each region of the segmentation tree
	Gradient method refers to the human-based segmentation (None) using the edges' map, or model-based (object used to explain) that 
	can be IG, BP, IXG and Saliency

	"""
	image =np.array(image).astype(np.float32)/255
	size = image.shape[:2]

	preprocess_transform = get_preprocess_transform(size_image)

	logits = net(preprocess_transform(image).unsqueeze(0).to(device)).detach().cpu().numpy()

	label_class = int(logits.argmax(axis=1)[0])
	
	#for human-based segmentation
	if gradient_method==None:
		gradient_image = detector.detectEdges(image)
	
	else: #for model-based segmentation
		explain_obj = eval(gradient_method)(net)
		attr_explain = attribute_image_features(net,explain_obj,label_class, preprocess_transform(image).unsqueeze(0).to(device), baselines=preprocess_transform(np.zeros((size_image,size_image,3),dtype=np.float32)).unsqueeze(0).to(device)) #, return_convergence_delta=True
		gradient_image = np.transpose(attr_explain.squeeze().cpu().detach().numpy().copy(), (1, 2, 0)).mean(axis=2)


	#construct graph from the weights of the image
	graph = hg.get_4_adjacency_graph(size)
	edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)


	#build initial tree T 
	t1b, altitudesb = buildTree(graph, edge_weights=edge_weights, areamin=areamin, tree=tree_ini,img_size = size_image,attribute = attribute)
	g = hg.CptHierarchy.get_leaf_graph(t1b)
	g_shape = hg.CptGridGraph.get_shape(g)
	edge_weights = hg.weight_graph(g, resize(image, g_shape, anti_aliasing=True), hg.WeightFunction.L2)

	
	#construct occlusion dataset
	dataset_occ = mask_nodes(t1b,altitudesb,image,areamin=areamin)
	dataset = dataset_masks(dataset_occ, transform=preprocess_transform, size_img = size_image)


	train_dataloader_occlusion = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
	
	#logits of the occluded images
	prob_idx = test_individual_feat_map.extract_probs(net, train_dataloader_occlusion, device = device)
	occ_nodes = []

	
	for prob_individual_mask in prob_idx:
		
		new_corr = (logits[0,label_class]- prob_individual_mask[label_class])
		occ_nodes.append(float(new_corr))

	occ_nodes = np.array(occ_nodes)

	return (occ_nodes - occ_nodes.min())/(occ_nodes.max() - occ_nodes.min()), t1b,altitudesb,g,edge_weights




def caoc_attribute(net,tree_ini,attribute,image,idx_img,rank_original,probs_original, labels, classes,size_image, complete_space = False,areamin= 500,device= 'cpu',gradient_method = None):
	"""
	CaOC attribute computation for each region of the segmentation tree
	Gradient method refers to the human-based segmentation (None) using the edges' map, or model-based (object used to explain) that 
	can be IG, BP, IXG and Saliency
	"""
	image =np.array(image).astype(np.float32)/255
	size = image.shape[:2]

	preprocess_transform = get_preprocess_transform(size_image)
	
	#for human-based segmentation
	if gradient_method==None:
		gradient_image = detector.detectEdges(image)
	else:  #for model-based segmentation
		explain_obj = eval(gradient_method)(net)
		label = net(preprocess_transform(image).unsqueeze(0).to(device)).detach().cpu().numpy()
		attr_explain = attribute_image_features(net,explain_obj,int(label.argmax(axis=1)[0]), preprocess_transform(image).unsqueeze(0).to(device), baselines=preprocess_transform(np.zeros((size_image,size_image,3),dtype=np.float32)).unsqueeze(0).to(device)) #, return_convergence_delta=True
		gradient_image = np.transpose(attr_explain.squeeze().cpu().detach().numpy().copy(), (1, 2, 0)).mean(axis=2)


	#construct graph from the weights of the image
	graph = hg.get_4_adjacency_graph(size)
	edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)

	#build initial tree T 
	t1b, altitudesb = buildTree(graph, edge_weights=edge_weights, areamin=areamin, tree=tree_ini,attribute = attribute)
	g = hg.CptHierarchy.get_leaf_graph(t1b)
	g_shape = hg.CptGridGraph.get_shape(g)
	edge_weights = hg.weight_graph(g, resize(image, g_shape, anti_aliasing=True), hg.WeightFunction.L2)

	#construct occlusion dataset
	dataset_occ = mask_nodes(t1b,altitudesb,image,areamin=areamin)
	dataset = dataset_masks(dataset_occ, transform=preprocess_transform, size_img = size_image)


	train_dataloader_occlusion = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
	
	#probabilites of the occluded images
	probs_modified = probs_original.copy()
	prob_idx = test_individual_feat_map.extract_probs(net, train_dataloader_occlusion, device = device)

	caoc_nodes = []

	for prob_individual_mask in prob_idx:
		probs_modified[idx_img,:] = prob_individual_mask  

		#--CaOC
		feat_map_probs_occlusion = order_functions.create_order(probs_modified, labels, classes,space=complete_space,all_samples = complete_space)      #create new order
		new_corr, signal = kendall_tau_correlation.count_changes(rank_original[1, :], feat_map_probs_occlusion[1, :], idx_img)	
		#--

		caoc_nodes.append(float(new_corr))
	caoc_nodes = np.array(caoc_nodes)

	return (caoc_nodes - caoc_nodes.min())/(caoc_nodes.max() - caoc_nodes.min()), t1b,altitudesb,g,edge_weights




def saliencyShaping(t1,alt1,g, attr,thr):
	"""
	Shaping method to incorporate the new attributes to the segmentation tree

	Credit to https://github.com/higra/Higra-Notebooks/
	Yongchao Xu, Edwin Carlinet, Thierry Géraud, Laurent Najman. Hierarchical Segmentation Using Tree-Based Shape Spaces.
	 IEEE Transactions on Pattern Analysis and Machine Intelligence, Institute of Electrical and Electronics Engineers,
	 2017, 39 (3), pp.457-469. ⟨10.1109/TPAMI.2016.2554550⟩. ⟨hal-01301966⟩
	"""
	
	sources, targets = t1.edge_list()
	g2 = hg.UndirectedGraph(t1.num_vertices() - t1.num_leaves())
	g2.add_edges(sources[t1.num_leaves():] - t1.num_leaves(), targets[t1.num_leaves():] - t1.num_leaves())

	# and compute the max tree
	t2 , a2 = hg.component_tree_max_tree(g2, attr)

	# Compute extinction values
	extinction = hg.attribute_dynamics(t2, a2)

	# The regions are the extrema of t2
	extrem = hg.attribute_extrema(t2, a2)

	extrema_nodes, = np.nonzero(extrem)

	# labelisations of vertices with extrema indices
	node_labels_t2 = np.zeros(t2.num_vertices(), dtype=np.int64)
	node_labels_t2[extrema_nodes] = extrema_nodes


	# labelisation of the leaves of t2 (ie, inner nodes of t1) by the index of the extrema they belong to (or 0 if there are not in an extrema)
	extrema_labels_t2 = hg.propagate_sequential(t2,  node_labels_t2, np.logical_not(extrem))[:t2.num_leaves()]

	# equivalent labelisation of the inner nodes of t1
	attr_extrema_labels_t1 = np.concatenate((np.zeros(t1.num_leaves(), dtype=np.int64), extrema_labels_t2))

	# There may be connected components of t1 with the same labels, we only keep  the largest node of each connected component,
	# ie. the one whose parent has a different label value
	attr_extrema_labels_t1[attr_extrema_labels_t1 == attr_extrema_labels_t1[t1.parents()]] = 0
	#filter_att = np.concatenate((np.zeros(t1.num_leaves(), dtype=np.int64), attr))
	
	

	# we replace non zero labels by the associated extinction value
	representent_nodes, = np.nonzero(attr_extrema_labels_t1)
	nodes_extinction = np.zeros_like(attr_extrema_labels_t1, dtype=np.float64)
	representent_nodes_labels = attr_extrema_labels_t1[representent_nodes]
	nodes_extinction[representent_nodes] = extinction[representent_nodes_labels]

	# maximal extinction on contours
	saliency_values = hg.propagate_sequential_and_accumulate(t1, nodes_extinction, hg.Accumulators.sum)
	saliency_values[saliency_values<saliency_values.max()*thr]=0
	leaf_weights = hg.reconstruct_leaf_data(t1, saliency_values, leaf_graph=None)


	return leaf_weights,nodes_extinction


	
