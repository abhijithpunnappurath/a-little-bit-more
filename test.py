# =============================================================================
#  @article{punnappurath2020bitdepth,
#    title={A little bit more: Bitplane-wise bit-depth recovery},
#    author={Punnappurath, Abhijith and Brown, Michael},
#    journal={},
#    year={},
#    volume={}, 
#    number={}, 
#    pages={}, 
#  }
# by Abhijith Punnappurath (05/2020)
# pabhijith@eecs.yorku.ca
# https://abhijithpunnappurath.github.io
# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
import numpy as np
from keras.models import load_model
from skimage.measure import compare_psnr, compare_ssim
import copy
import cv2

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    
    parser.add_argument('--set_names', default='Kodak', type=str, help='name of test dataset')
    parser.add_argument('--type_8_or_16', default='0', type=str, help='are images in the corresponding set_names folder 8 or 16 bit, 0 = 8 and 1 = 16')    
    
    parser.add_argument('--quant', default=4, type=int, help='bit position - quantization starts')
    parser.add_argument('--quant_end', default=8, type=int, help='bit position - quantization ends')
        
    parser.add_argument('--model_dir', default='pretrained_models', type=str, help='directory of the model')
    parser.add_argument('--dep',default=4,type=int,help='number of residual units')
    parser.add_argument('--model_name', default=[0,30,30,30,   30, 30, 30, 30,   30, 30, 30, 30,  30, 30, 30, 30], type=list, help='the model epoch number')    
    
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=0, type=int, help='1 or 0, 1 = save image and PSNR/SSIM values')
    return parser.parse_args()
    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return img[np.newaxis,...]

def from_tensor(img):
    return np.squeeze(img)

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)
     
def remove_zero_bits(img,bitplane):
    binval = np.unpackbits(img.byteswap().view('uint8'))
    binval = np.reshape(binval,[binval.size//16,16])
    binval = np.concatenate((np.zeros([np.size(binval,0),16-bitplane]).astype('uint16'),binval[:,0:bitplane]),axis=1)
    
    tempvar = copy.copy(binval)
    binval[:,0:8]=tempvar[:,8:16]
    binval[:,8:16]=tempvar[:,0:8]
    
    binval = np.packbits(binval).view('uint16')
    binval = np.reshape(binval,img.shape)
    
    return binval

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        cv2.imwrite(path,result)

if __name__ == '__main__':    
    
    args = parse_args()
    
    type_8_or_16_list = [int(item) for item in args.type_8_or_16.split(',')]
    set_names_list = [item for item in args.set_names.split(',')]
    
    
    model=[[] for k in range(args.quant_end)]
    for i in range(args.quant,args.quant_end):
        model[i]=load_model(os.path.join(args.model_dir, 'D' + str(args.dep),'bitpos_' + '{:0>2d}'.format(i+1), 'model_'+'{:0>3d}'.format(args.model_name[i])+'.hdf5'),compile=False)
        log('load trained model - {:0d} - {:0>3d}'.format(i+1,args.model_name[i]))
        
    for folder_index,set_cur in enumerate(set_names_list):  
        
        if args.save_result:
            if not os.path.exists(os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur)):
                os.makedirs(os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur))
                       
        psnrs = []
        ssims = [] 
        
        for im in os.listdir(os.path.join(args.set_dir,set_cur)): 
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = cv2.imread(os.path.join(args.set_dir,set_cur,im),-1)
                
                if(type_8_or_16_list[folder_index]):
                
                    binval = x
                    binval = np.unpackbits(binval.byteswap().view('uint8'))
                    binval = np.reshape(binval,[binval.size//16,16])
                    binval_input = copy.copy(binval)
                    binval_gt = copy.copy(binval)
                    binval_input = np.concatenate((binval[:,0:args.quant],np.zeros([np.size(binval,0),16-args.quant]).astype('uint16')),axis=1)
                    binval_gt = np.concatenate((binval[:,0:args.quant_end],np.zeros([np.size(binval,0),16-args.quant_end]).astype('uint16')),axis=1)
                
                else:
                    binval = x
                    binval = np.unpackbits(binval)
                    binval = np.reshape(binval,[binval.size//8,8])
                    binval_input = copy.copy(binval)
                    binval_gt = copy.copy(binval)
                    binval_input = np.concatenate((binval[:,0:args.quant],np.zeros([np.size(binval,0),8-args.quant]).astype('uint16'),np.zeros([np.size(binval,0),8]).astype('uint16')),axis=1)
                    binval_gt = np.concatenate((binval[:,0:args.quant_end],np.zeros([np.size(binval,0),8-args.quant_end]).astype('uint16'),np.zeros([np.size(binval,0),8]).astype('uint16')),axis=1)
                    
                                
                # adjust for 16 bit
                tempvar = copy.copy(binval_input)
                binval_input[:,0:8]=tempvar[:,8:16]
                binval_input[:,8:16]=tempvar[:,0:8]
                tempvar = copy.copy(binval_gt)
                binval_gt[:,0:8]=tempvar[:,8:16]
                binval_gt[:,8:16]=tempvar[:,0:8]                
                
                binval_input = np.packbits(binval_input).view('uint16')
                binval_gt = np.packbits(binval_gt).view('uint16')
                binval_input = np.reshape(binval_input,x.shape)
                binval_gt = np.reshape(binval_gt,x.shape)
                                                
                x = binval_gt.astype('float32')/65535.0
                y = binval_input.astype('float32')/65535.0
                                          
                start_time = time.time()
                quant_counter=args.quant
                x_=to_tensor(y)
                
                for i in range(args.quant,args.quant_end):
                    xr_ = model[i].predict(x_) # inference
                    x_ = x_ + (xr_)* (2**(16-quant_counter-1)/65535)                    
                    quant_counter=quant_counter+1
                
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                
                x_ = from_tensor(x_)
                
                x = (65535*x).astype('uint16')
                y = (65535*y).astype('uint16')
                x_ = (65535*x_).astype('uint16')                
                
                x=remove_zero_bits(x,args.quant_end)
                x_=remove_zero_bits(x_,args.quant_end)
                y=remove_zero_bits(y,args.quant_end) 
                
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    save_result(x_,path=os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur,name+'_output.png')) # save the result
#                    save_result(x,path=os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur,name+'_gt.png')) # save the ground truth
#                    save_result(y,path=os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur,name+'_input.png')) # save the zp input
                
                psnr_x_ = compare_psnr(x, x_,data_range=2**args.quant_end-1)
                ssim_x_ = compare_ssim(x, x_,multichannel=True,data_range=2**args.quant_end-1)
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
                
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        
        if args.save_result:
            save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur,'results.txt'))
            save_result(np.hstack((psnr_avg,ssim_avg)),path=os.path.join(args.result_dir,'D' + str(args.dep) + '_quant_' + str(args.quant) + '_' + str(args.quant_end),set_cur,'results_avg.txt'))
            
        log('Dataset: {0:10s} \n  PSNR = {1:2.4f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
        
        
        


