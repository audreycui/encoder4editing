from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import data_utils
from utils.parallelfolder import ParallelImageFolders

import torch
import os
import glob
import numpy as np


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		#self.source_paths = sorted(data_utils.make_dataset(source_root))
		#self.target_paths = sorted(data_utils.make_dataset(target_root))
		print('source_root: ' + source_root) 
		source_path_file = os.path.join(source_root, source_root[source_root.find('/')+1:] + '_list.txt')   
		print(source_path_file)
		#self.source_paths = glob.glob(os.path.join(source_root,'/**/*.webp'), recursive = True)
		with open(source_path_file) as f:
			self.source_paths = f.read().splitlines()
		self.source_paths = [os.path.join(source_root, s) for s in self.source_paths]            
		print('LEN PATHS: ' + str(len(self.source_paths)))
        
		target_path_file = os.path.join(target_root, target_root[target_root.find('/')+1:] + '_list.txt')      
		#self.source_paths = ParallelImageFolders(target_root)     
		#self.target_paths = glob.glob(os.path.join(target_root,'/**/*.webp'), recursive = True)
		with open(target_path_file) as f:
			self.target_paths = f.read().splitlines()
		self.target_paths = [os.path.join(target_root, s) for s in self.target_paths]                        
		print('LEN PATHS: ' + str(len(self.target_paths)))
        
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im

class BedroomsDataset(Dataset):
    # need to define your __init__()
    def __init__(self): 
        super(BedroomsDataset, self).__init__()
        self.data_root = '/data/vision/torralba/datasets/LSUN/image/bedroom_val'
        self.img_list = glob.glob(os.path.join(self.data_root,'*.webp'))
        self.data_length = len(self.img_list)
        
    # need to define your __len__()
    def __len__(self):
        return self.data_length
    
    # need to define your __get_item__()
    def __getitem__(self, idx):
        im_name = self.img_list[idx]
        image = Image.open(im_name)
        # need 1) make it [-1,1] 2) HWC [100,100,3] ==> CHW [3, 100, 100], send me np (or tensor)
        image = image.crop((0, 0, 256, 256)) 
        image = np.array(image)
        image = (image/255 - 0.5) * 2
        image = np.transpose(image, (2,0,1))
        image = torch.tensor(image, dtype=torch.float32)
        

        return [image]
#        return [z_vect, image, mask, target]
