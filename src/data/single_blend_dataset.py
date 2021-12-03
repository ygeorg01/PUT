from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from data.augmentations import panorama_augmentation
import torch
import numpy as np
import cv2
import os


class SingleBlendDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # self.A_path =  opt.dataroot
        # print('Consistency dataroot: ', opt.consistency_dataroot)
        self.dir_path = opt.dataroot
        self.A_paths = sorted(os.listdir(self.dir_path))
        print('A paths: ', self.A_paths, self.dir_path)

        # self.B_path = opt.consistency_dataroot
        # self.B_path = opt.results_dir
        # self.mask_path = opt.mask_path
        # self.mask_path = opt.mask_path

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # print('Input nc: ', input_nc == 1, input_nc)
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.transform = get_transform(opt, grayscale=False)
        self.transform_gray = get_transform(opt, grayscale=False)
        self.transform_mask = get_transform(opt, grayscale=True, mask=True)
        self.kernel = torch.ones((3, 3, 3))

    def __getitem__(self, index):

        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing.
s
        Returns a dictionary that contains A and A_paths.
            A(tensor) - - an image in one domain.
            A_paths(str) - - the path of the image.
        """

        # A_path = self.A_path+'/'+str(index).zfill(5)+'.
        print('Dir path: ', self.dir_path)
        A_path = os.path.join(self.dir_path, self.A_paths[index])
        print('Image Path: ', A_path, index)
        out_path = os.path.join(self.dir_path[:-6]+'output/', self.A_paths[index])
        print('Output path: ', out_path)
        mask_path = './mask_eval.png'
        # if self.B_path != None:
        # B_path = self.B_path
        # B_path = B_path+A_path.split('/')[-1]
        # mask_path = self.mask_path

        # print('Mask path: ', mask_path)

        A_img = Image.open(A_path)  # .convert('RGB')
        out_img = Image.open(out_path)
        mask = Image.open(mask_path)
        # if self.B_path != None:
        # B_img = Image.open(B_path).convert('RGB')
        # mask = Image.open(mask_path).convert('RGB')
        # mask2 = Image.open('/home/vcg/Desktop/Urban Enviroment Understanding/Mesh_Texturing/Scenes/scene4/consistency/mask/mask_12.png').convert('RGB')

        # mask = Image.open(mask_path).convert('RGB')

        # if self.B_path != None:

        # B = self.transform(B_img)#[[2,1,0],:,:]
        # mask = self.transform_mask(mask).round().int()
        # mask = torch.cat([mask, mask, mask], dim=0)

        # extra part
        # mask = torch.nn.functional.conv2d(mask.unsqueeze(0).int(), self.kernel.unsqueeze(0).int(), padding=(1, 1))
        # mask = torch.stack((mask, mask, mask), dim=2).squeeze()

        # B[mask!=27] = -1
        #
        # B = self.apply_mask(B, mask)
        # B_t = B.clone()

        A_tensor = self.transform(A_img)
        out_tensor = self.transform(out_img)
        # print('Out_tensor shape: ', out_tensor[])
        # Change pattern
        out_tensor = out_tensor + out_tensor*0.2
        mask_tensor = self.transform(mask)

        mask_tensor[A_tensor==-1] = -1
        A_tensor[mask_tensor==1] = out_tensor[mask_tensor==1]



        # if self.B_path != None:
        # B[A_or==-1] = -1
        # B[B == -1] = A_or[B == -1]
        # A=B
        # else:
        A = A_tensor

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # print('lENGTH OF a: ', len(self.A_path))
        return len(self.A_paths)

    def apply_mask(self, source_image, mask):

        # source_image = source_image*mask
        source_image[mask == -1] = -1

        return source_image

    def tensor2im(self, input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.

        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
        # print('Input image: ', input_image.shape)
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            image_numpy = image_numpy[:, :, [2, 1, 0]]
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image

        return image_numpy.astype(imtype)
