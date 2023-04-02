import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # path to A domain image
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # path to B domain image
        self.dir_gt_A_cell = os.path.join(opt.dataroot, opt.phase + 'A_cell')  # path to A domain cell nuclei anotation
        self.dir_gt_B_cell = os.path.join(opt.dataroot, opt.phase + 'B_cell')  # path to B domain cell nuclei anotation
        self.dir_gt_A_line = os.path.join(opt.dataroot, opt.phase + 'A_layer') # path to A domain skin layers anotation
        self.dir_gt_B_line = os.path.join(opt.dataroot, opt.phase + 'B_layer') # path to B domain skin layers anotation

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.gt_A_cell_paths = sorted(make_dataset(self.dir_gt_A_cell, opt.max_dataset_size))  
        self.gt_B_cell_paths = sorted(make_dataset(self.dir_gt_B_cell, opt.max_dataset_size))
        self.gt_A_line_paths = sorted(make_dataset(self.dir_gt_A_line, opt.max_dataset_size))  
        self.gt_B_line_paths = sorted(make_dataset(self.dir_gt_B_line, opt.max_dataset_size))
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_img = get_transform(self.opt, grayscale=False)
        self.transform_gt = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        gt_A_cell_path = self.gt_A_cell_paths[index % self.A_size]
        gt_A_line_path = self.gt_A_line_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        gt_B_cell_path = self.gt_B_cell_paths[index_B]
        gt_B_line_path = self.gt_B_line_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        gt_A_cell_img = Image.open(gt_A_cell_path).convert('L')
        gt_B_cell_img = Image.open(gt_B_cell_path).convert('L')
        gt_A_line_img = Image.open(gt_A_line_path).convert('L')
        gt_B_line_img = Image.open(gt_B_line_path).convert('L')
        # apply image transformation
        A = self.transform_img(A_img)
        B = self.transform_gt(B_img)
        A_gt_cell = self.transform_gt(gt_A_cell_img)
        B_gt_cell = self.transform_gt(gt_B_cell_img)
        A_gt_line = self.transform_gt(gt_A_line_img)
        B_gt_line = self.transform_gt(gt_B_line_img)

        return {'A': A, 'B': B, 
                'A_gt_cell': A_gt_cell, 'B_gt_cell': B_gt_cell,
                'A_gt_line': A_gt_line, 'B_gt_line': B_gt_line, 
                'A_paths': A_path, 'B_paths': B_path, 
                'gt_A_cell_path': gt_A_cell_path, 'gt_B_cell_path': gt_B_cell_path,
                'gt_A_line_path': gt_A_line_path, 'gt_B_line_path': gt_B_line_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
