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
import torchvision.transforms as transforms
np.set_printoptions(threshold=sys.maxsize)
from argparse import ArgumentParser
from PIL import Image
import pickle
import matplotlib.pyplot as plt

import utils_caoc.order_functions as order_functions
import utils_model.test_individual_feat_map as test_individual_feat_map
import utils_model.load_models as load_models
from tree_ import saliencyShaping,caoc_attribute,occ_attribute


def pixel_impact_rate(ini,fim,sum_matrix):

	label_ini = np.argmax(ini, axis=1)[0]
	print(ini, fim, sum_matrix)
	
	diff = np.abs(ini[0,label_ini] - fim[0,label_ini])

	
	if not sum_matrix == 0:
	
		diff = diff/sum_matrix
	else:
		diff = 0
			

	return diff,ini[0,label_ini] - fim[0,label_ini]



def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

def get_preprocess_transform(size_img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((size_img, size_img)),
	normalize
    ])    

    return transf



if __name__ == "__main__":



	parser = ArgumentParser()

	parser.add_argument("--model", type=str, default="vgg")
	parser.add_argument("--dataset", type=str, default="cat_dog")
	parser.add_argument("--num_images", type=int, default=512)
	parser.add_argument("--size_image", type=int, default=224)
	parser.add_argument("--idx_img", type=int, default=0)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--num_threads", type=int, default=4)
	parser.add_argument("--thr_viz", type=float, default=0.75)
	parser.add_argument("--gpu_id", type=str, default="0")
	parser.add_argument("--tree_type", type=str, default="watershed")
	parser.add_argument("--attribution_ini", type=str, default="area")
	parser.add_argument("--attribution_occ", type=str, default="occ")
	parser.add_argument("--min_region", type=int, default=500)
	parser.add_argument("--seg_type", type=str, default="")
	parser.add_argument("--inclusion", type=int, default=0)
	parser.add_argument("--classes", type=int, default=2)

	parser.set_defaults(feature=False)
	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	WORKERS = args.num_workers
	NUM_THREADS = args.num_threads
	name_net = args.model
	dataset_name = args.dataset
	size_image = args.size_image
	classes = args.classes
	num_images = args.num_images
	thr = args.thr_viz
	inclusion = args.inclusion
	tree_type = args.tree_type
	attribution_ini = args.attribution_ini
	attribution_occ = args.attribution_occ
	min_region = args.min_region
	seg_type = args.seg_type
	idx_img = args.idx_img

	
	np.random.seed(seed=0)
	torch.manual_seed(0)

	if args.gpu_id == "-1":
	    device = "cpu"

	elif torch.cuda.is_available():  
	    device = "cuda:"+str(args.gpu_id) 
	    torch.cuda.set_device(device)
	else:  
	    device = "cpu"  

	
	torch.multiprocessing.set_sharing_strategy('file_system')

	weights_model = ''
	path_label_file = ''
	path_images = ''
	name_label_file = ''


	net, path_label_file,path_images,name_label_file,analyzed_layer= load_models.load_models(name_net, dataset_name,classes)
	'''	
	if dataset_name == 'cat_dog':
		path_images = 'datasets/cat_dog/'
		net.load_state_dict(torch.load('../checkpoints/cat_dog_'+name_net+'_weights.pth',map_location='cuda'))
	if dataset_name == 'cifar10':
		path_images = 'datasets/cifar10/'
		net.load_state_dict(torch.load('checkpoints/cifar10/'+name_net+'_weights.pth'))
		
	'''
	if dataset_name == 'cat_dog':
		path_images = '../../../Trainings/binary_task_cat_dog/data/train/'
		path_label_file = '../'+path_label_file
		net.load_state_dict(torch.load('../checkpoints/cat_dog_'+name_net+'_weights.pth',map_location='cuda'))
	if dataset_name == 'cifar10':
		path_images = '../datasets/cifar10/'
		path_label_file = '../'+path_label_file
		net.load_state_dict(torch.load('../checkpoints/cifar10/'+name_net+'_weights.pth'))
		'''
		checkpoint = torch.load('../../../RIG_tests/cifar10/vgg16_bn/version_1/checkpoints/epoch=93-step=18423.ckpt')
		print(checkpoint)
		net = test_individual_feat_map.CIFAR10Module(args)
		net.load_state_dict(checkpoint['state_dict'])
		net = net.model
		torch.save(net.state_dict(), '../checkpoints/cifar10/'+name_net+'_weights.pth')
		exit()
		'''
		#net.load_state_dict(checkpoint.model)
	net = net.to(device)
	net.eval()


	preprocess_transform = get_preprocess_transform(size_image)


	if dataset_name == 'cifar10':
		train_dataset = test_individual_feat_map.CIFAR10Data(path_images,  BATCH_SIZE, WORKERS, False).get_dataset_val(transform=preprocess_transform)
		train_dataset2 = test_individual_feat_map.CIFAR10Data(path_images,  BATCH_SIZE, WORKERS, False).get_dataset_val()

	#cat_dog
	else:	
		final_to_intercalate = pd.read_csv(path_label_file+name_label_file)
		train_dataset = test_individual_feat_map.MyDataset(path_label_file,path_images,name_label_file,num_images,transform=preprocess_transform)


		label = 1-np.array(final_to_intercalate['label'].tolist())[:num_images] 



	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=False,num_workers=WORKERS)
	probs_original = test_individual_feat_map.extract_probs(net, train_dataloader,device = device)	

	labels = np.argmax(probs_original, axis=1)[:num_images] 

	rank_original = order_functions.create_order(probs_original,labels, classes,space = False,all_samples = False)	



	save_path_img = 'outputs/tree_'+attribution_occ+'/'+seg_type+'_'

	explain_method = seg_type
	if explain_method == '':
		explain_method = None
	
	attribute = 'hg.watershed_hierarchy_by_'+attribution_ini #only used if it is watershed 

	
	###############################################################################################################################
	

	print(min_region, idx_img,tree_type,attribution_ini,seg_type)
	

	if dataset_name == 'cifar10':
		sample1,label =train_dataset2.__getitem__(idx_img)
		
		sample1 = sample1.resize((size_image,size_image))
	else:
		path_img = path_images+final_to_intercalate['path'][idx_img]
		sample1 = get_image(path_img).resize((size_image,size_image))


	logit_ini = net(preprocess_transform(sample1).unsqueeze(0).to(device)).detach().cpu().numpy()
	label = int(np.argmax(logit_ini, axis=1)[0])
	

	if attribution_occ =='caoc':
		att_nodes,t1,altitudes1,g,edge_weights= caoc_attribute(net,tree_type,attribute,sample1,idx_img,rank_original,probs_original, labels, [labels[idx_img]],size_image,complete_space = False,areamin = min_region, device= device,gradient_method = explain_method)

	elif attribution_occ =='occ':

		att_nodes,t1,altitudes1,g,edge_weights=occ_attribute(net,tree_type,attribute,sample1,idx_img,size_image, complete_space = False,areamin= min_region,device= device,gradient_method = explain_method)
	else:
		print('Occlusion metric not defined!')
		exit()
	
	matrix_weights,_ = saliencyShaping(t1,altitudes1,g, att_nodes,0.0)

	

	matrix_weights[matrix_weights<(matrix_weights.max()*thr)] = 0
	matrix_weights[matrix_weights>0] = 1
	

	sample2 = np.array(sample1)
	
	if inclusion == 1: #only the important components
		sample2[matrix_weights==0] = 0
		plt.imshow(sample2)
		plt.savefig(save_path_img+'regions'+str(min_region)+'_inclusion_image_idx' + str(idx_img)+'_model_'+name_net+'.png')
	else: #occlusion of the important component
		sample2[matrix_weights==1] = 0
		plt.imshow(sample2)
		plt.savefig(save_path_img+'regions'+str(min_region)+'_exclusion_image_idx' + str(idx_img)+'_model_'+name_net+'.png')

		

	
	sample2 = Image.fromarray(np.uint8(sample2))


	logit_end = net(preprocess_transform(sample2).unsqueeze(0).to(device)).detach().cpu().numpy()

	print('PIR and Impact occlusion on the model:', pixel_impact_rate(logit_ini,logit_end,matrix_weights.sum()))

