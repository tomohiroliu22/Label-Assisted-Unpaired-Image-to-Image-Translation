import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  
        self.dir_A_gt_cell = os.path.join(opt.dataroot, opt.phase + 'A_gt_cell')
        self.dir_A_gt_line = os.path.join(opt.dataroot, opt.phase + 'A_gt_line') 
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.gt_A_cell_paths = sorted(make_dataset(self.dir_A_gt_cell, opt.max_dataset_size))
        self.gt_A_line_paths = sorted(make_dataset(self.dir_A_gt_line, opt.max_dataset_size))
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform_img = get_transform(self.opt, grayscale=False)
        self.transform_gt = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_paths = self.A_paths[index]
        gt_A_cell_paths = self.gt_A_cell_paths[index]
        gt_A_line_paths = self.gt_A_line_paths[index]
        
        w, h = 256, 256
        A = Image.open(A_paths).convert('RGB').resize((w,h))
        gt_A_cell = Image.open(gt_A_cell_paths).convert('L').resize((w,h))
        gt_A_line = Image.open(gt_A_line_paths).convert('L').resize((w,h))
        gt_A = np.array(gt_A_line)
        gt_A[np.array(gt_A_cell) == 255] = 255
        gt_A = Image.fromarray(gt_A)
        A = self.transform_img(A)
        gt_A = self.transform_gt(gt_A)

        return {'A': A, 'gt_A': gt_A, 'A_paths': A_paths, 'gt_A_cell_paths': gt_A_cell_paths, 'gt_A_line_paths': gt_A_line_paths}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
