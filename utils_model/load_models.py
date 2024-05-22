import torch
import torchvision.models as models
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from deepface import DeepFace
from cifar10_models.resnet import resnet18
from cifar10_models.vgg import vgg16_bn

def load_models(name_net, dataset_name,classes,pre_dataset = None, weights=None):


	weights_model = ''
	path_label_file = ''
	path_images = ''
	name_label_file = ''
	analyzed_layer = ''


	#--initializing model
	if name_net == 'vgg':
		analyzed_layer = 'features[28]'
		net = models.vgg16(pretrained=True)
		dim_in = net.classifier[6].in_features
		net.classifier[6] = nn.Linear(dim_in, classes)

		if dataset_name == 'cifar10':
			net = vgg16_bn()

	elif name_net == 'resnet':
		analyzed_layer = 'layer4[1].conv2'
		net = models.resnet18(pretrained=True)
		dim_in = net.fc.in_features
		net.fc = nn.Linear(dim_in, classes)

		if dataset_name == 'cifar10':
			net = resnet18()

	else:
		print('Model not included here!')
		exit()
	#--


	#--dataset parameters
	
			
	weights_model = 'checkpoints/'+dataset_name+'_'+name_net+'_weights.pth'
	path_label_file = 'datasets/'+dataset_name+'/'
	path_images = 'datasets/'+dataset_name+'/images/'
	name_label_file = 'intercalate_images_train.csv'

	if not weights == None:
		weights_model = 'checkpoints/'+weights

		
	#--

	try:
		
		net.load_state_dict(torch.load(weights_model),strict=True) #load weights
		print(weights_model)

	except:
		pass

	return net, path_label_file,path_images,name_label_file, analyzed_layer


