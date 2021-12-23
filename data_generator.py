# =============================================================================
#  @misc{punnappurath2020little,
#    title={A Little Bit More: Bitplane-Wise Bit-Depth Recovery},
#    author={Abhijith Punnappurath and Michael S. Brown},
#    year={2020},
#    eprint={2005.01091},
#    archivePrefix={arXiv},
#    primaryClass={eess.IV}
#    } 
# by Abhijith Punnappurath (05/2020)
# pabhijith@eecs.yorku.ca
# https://abhijithpunnappurath.github.io
# =============================================================================

# no need to run this code separately

import glob
import cv2
import numpy as np

patch_size, stride = 48, 48
aug_times = 1

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name,-1)  # -1 => any depth, in this case 16 
    h, w, cc = img.shape
    patches = []
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size,:]
            # data aug
            for k in range(0, aug_times):
                x_aug = data_aug(x, mode=np.random.randint(0,8))
                patches.append(x_aug)
            
    return patches

def datagenerator(data_dir='data/train',batch_size=128,verbose=False):
    
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)        
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done')
    data = np.concatenate(data)
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data

if __name__ == '__main__':   

    data = datagenerator(data_dir='data/train')