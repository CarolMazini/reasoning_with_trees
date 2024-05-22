#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy
import torchvision.transforms as T
import h5py
import tensorflow as tf
import tensorflow.keras as k
import pytorch_lightning as pl
import requests
from torchvision.datasets import CIFAR10


from cifar10_models.resnet import resnet18
from cifar10_models.vgg import vgg16_bn
from cifar10_models.schduler import WarmupCosineLR
from pytorch_lightning.metrics import Accuracy

all_classifiers = {
    "vgg": vgg16_bn(),
    "resnet": resnet18()
}


class CIFAR10Module(pl.LightningModule):
    """
    Credit to https://github.com/huyvnphan/PyTorch_CIFAR10 (MIT license)
    Huy Phan. (2021). huyvnphan/PyTorch_CIFAR10 (v3.0.1). Zenodo. https://doi.org/10.5281/zenodo.4431043
    """
    def __init__(self,hparams=None):
        super().__init__()
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers['vgg']

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return predictions,loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        _,loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        _,loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        _,loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=5e-7,
            weight_decay=5e-5,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = 500 * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


########################################################
##                   DATASET CIFAR10                  ##	
########################################################


class CIFAR10Data(pl.LightningDataModule):
	"""
	Credit to https://github.com/huyvnphan/PyTorch_CIFAR10 (MIT license)
	Huy Phan. (2021). huyvnphan/PyTorch_CIFAR10 (v3.0.1). Zenodo. https://doi.org/10.5281/zenodo.4431043
	"""
	def __init__(self,data_dir, batch_size, num_workers, shuffle):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.shuffle = shuffle
		self.mean = (0.4914, 0.4822, 0.4465)
		self.std = (0.2471, 0.2435, 0.2616)


	def download_weights():
		url = (
			"https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
		)

		# Streaming, so we can iterate over the response.
		r = requests.get(url, stream=True)

		# Total size in Mebibyte
		total_size = int(r.headers.get("content-length", 0))
		block_size = 2 ** 20  # Mebibyte
		t = tqdm(total=total_size, unit="MiB", unit_scale=True)

		with open("state_dicts.zip", "wb") as f:
			for data in r.iter_content(block_size):
				t.update(len(data))
				f.write(data)
		t.close()

		if total_size != 0 and t.n != total_size:
			raise Exception("Error, something went wrong")

		print("Download successful. Unzipping file...")
		path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
		directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
		with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
			zip_ref.extractall(directory_to_extract_to)
			print("Unzip file successful!")

	def train_dataloader(self):
		transform = T.Compose(
			[
				T.RandomCrop(32, padding=4),
				T.RandomHorizontalFlip(),
				T.ToTensor(),
				T.Normalize(self.mean, self.std),
			]
		)
		dataset = CIFAR10(root=self.data_dir, train=True, transform=transform,download=True)
		dataloader = DataLoader(
			dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=self.shuffle,
			drop_last=False,
			pin_memory=True,
		)
		print('Correctly load train!!!')
		return dataloader

	def val_dataloader(self):
		transform = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(self.mean, self.std),
			]
		)
		dataset = CIFAR10(root=self.data_dir, train=False, transform=transform,download=True)
		dataloader = DataLoader(
			dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=False,
			pin_memory=True,
		)
		print('Correctly load val!!!')
		return dataloader

	def test_dataloader(self):
		return self.val_dataloader()

	def get_dataset_train(self, num_images = None,type_img=None, new_compl_path = None):
		transform = T.Compose(
			[
				T.RandomCrop(32, padding=4),
				T.RandomHorizontalFlip(),
				T.ToTensor(),
				T.Normalize(self.mean, self.std),
			]
		)
		dataset = CIFAR10(root=self.data_dir, train=True, transform=transform,download=False)
		
		print('Correctly load train!!!')
		return dataset


	def get_dataset_val(self, num_images = None,type_img=None, new_compl_path = None, transform=None):
		transform2 = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(self.mean, self.std),
			]
		)
		dataset = CIFAR10(root=self.data_dir, train=False, transform=transform,download=True)
		
		print('Correctly load val!!!')
		return dataset




########################################################
##               DATASET CAT_DOG                      ##	
########################################################

class MyDataset(Dataset):
	def __init__(self,base,base_image,input_path, num_images = None, transform=None, idx_change = None,image_modify = None):
		
		
		self.input_samples = pd.read_csv(base+input_path)
		self.input_samples = self.input_samples.loc[self.input_samples['label'] != -1].reset_index()
		self.input_samples_param = self.input_samples
		self.transform = transform
		self.base = base
		self.base_image = base_image
		self.num_images = num_images
		self.image_modify = image_modify
		self.idx_change = idx_change

	def __getitem__(self, idx):
		param = 0
		image_path = self.input_samples['path'][idx]
		#print(self.base_image,image_path)
		label = self.input_samples['label'][idx]

		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		
		sample = sample.resize((224,224))
	
		#train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		#train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)

		
		#sample = (sample - train_mean) / train_std
		if (self.image_modify != None) and (self.idx_change== idx):

			sample = self.image_modify.permute(1,2,0).detach().cpu().numpy()
			
		
		label = int(label)
		if label==1:
			
			label =  [1,0]
		else:
			
			label =  [0,1]

		if self.transform:
			sample = self.transform(sample)
			
		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):
		if self.num_images == None:
			return len(self.input_samples_param)
		else:
			return self.num_images





#********************************************************************************************************************#

def test(net, dataloader,device='cpu'):

	'''
	Input: model and dataloader with images to be tested
	Output: total accuracy and loss
	'''

	correct = 0
	total = 0
	correct_cat = 0
	dog_label = 0
	cat_label = 0
	correct_dog = 0
	loss = 0
	prob = nn.Softmax(dim = 1)
	net.eval()
	with torch.no_grad():
		for i,data in enumerate(dataloader):
			print('Batch:', i)
			images,_, labels = data
			images, labels = images.to(device), labels.to(device)

			outputs = prob(net(images))
			loss +=  torch.nn.functional.binary_cross_entropy(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			_, labels_aux = torch.max(labels, 1)
			total += labels.size(0)
			correct += (predicted == labels_aux).sum().item()
			dog_label += labels_aux.sum()
			cat_label += len(labels_aux) - labels_aux.sum()
			for t in range(len(predicted)):
			
				correct_dog += ((predicted[t] == labels_aux[t]) and (labels_aux[t] == 1)).sum().item()
				correct_cat += ((predicted[t] == labels_aux[t]) and (labels_aux[t] == 0)).sum().item()


	print(dog_label)
	print('Loss val:', loss/i)
	print('Accuracy: %f %%' % (100 * correct / total))
	
	print(correct_dog,correct_cat, dog_label, cat_label)

	return (100 * correct / total), (loss.detach().cpu().numpy()/i)


def extract_probs(net, dataloader, idx=None,device='cpu'):

	'''
	Input: model and dataloader with images to extract activations
	Output: matrix of extracted final probabilities (before softmax)
	'''

	activations = []
	
	net.eval()
	with torch.no_grad():
		if idx==None:
			for i,data in enumerate(dataloader):
				print('Batch:', i)
				try:
					images,_, labels = data
				except:
					images, labels = data
				images, labels = images.to(device), labels.to(device)

				outputs = net(images)
				activations.extend(np.array(outputs.detach().cpu()))
		else:
			image,_, labels = dataloader.dataset[idx]
			outputs = net(torch.Tensor(image).unsqueeze(0).cuda())
			activations = np.array(outputs.detach().cpu())

	
	return np.array(activations)




def zera(feat_maps, num_feat,analyzed_layer, model):

	'''
	Input: list of feature maps ids to maintain, number of feature maps, analyzed layer to filter the file, inverted top with the chosen number of smallest correlations to be deleted, and the copy of the model to be manipulated
	Output: the modified copy of the model
	'''

	## testar isso aqui, nao sei se ta certo
		
	string ='model.'+analyzed_layer

	with torch.no_grad():
		#print(m['layer'])
		module = eval(string)
		#print(num_feat)
		#exit()

		for feat_map in range(num_feat):
			#print(module)
			if feat_map not in feat_maps:
				module.weight[feat_map,:,:,:] = torch.nn.Parameter(torch.zeros_like(module.weight[feat_map,:,:,:]))
			

	return(model)



