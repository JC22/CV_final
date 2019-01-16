import numpy as np
import argparse
import cv2
import time
import torch
import torch.nn as nn
import pdb
#from models import *
from torch.autograd import Variable
from util import writePFM
from mc_cnn.match import get_cost_volume 
from stackhourglass import *
from image_rectification import rectify
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')

def filler(Il, left_disp, right_disp, max_disp):
    h, w, ch = Il.shape
    Y = np.zeros((h,w))
    for i in range(h) :
        Y[i] = np.ones(w)*(i)
    
    X = np.zeros((h,w))
    for j in range(h) :
        X[j] = (np.array([x for x in range(w)]))
    
    X = X - left_disp
    X[X<1] = 1

    index = [[Y[i][j], X[i][j]] for i in range(h) for j in range(w)]
    index = np.array(index, dtype=int)
    index_new = np.array(X, dtype=int)
    new_label = left_disp
    for i in range(h) :
        for j in range(w) :
            index_new[i][j] = right_disp[index[i*w+j][0], index[i*w+j][1]]
    new_label[abs(left_disp -index_new) >= 1] = -1    

    numDis = max_disp
    input_labels = new_label
    h, w = input_labels.shape
    pix = np.zeros((h, w))
    pix[input_labels < 0] = 1
    fill_value = np.ones((h)) * numDis
    final_labels_filled = input_labels
    for i in range(w) :
        curCol = input_labels[:, i].copy()
    
        for j in range(len(curCol)) :
            if curCol[j] == -1 :        
                curCol[j] = fill_value[j]
        for k in range(len(curCol)) :
            if curCol[k] != -1 :        
                fill_value[k] = curCol[k]
    
        final_labels_filled[:, i] = curCol
    
    fill_value = np.ones((h)) * numDis
    final_labels_filled1 = input_labels

    for i in range(w-1, -1 , -1) :
        curCol = input_labels[:, i].copy()
        for j in range(len(curCol)) :
            if curCol[j] == -1 :        
                curCol[j] = fill_value[j]
        for k in range(len(curCol)) :
            if curCol[k] != -1 :        
                fill_value[k] = curCol[k]
        final_labels_filled1[:, i] = curCol
    
    final_labels = np.minimum(final_labels_filled, final_labels_filled1)
    # result = cv2.ximgproc.weightedMedianFilter(Il,final_labels,)
    result = cv2.ximgproc.weightedMedianFilter(Il.astype('uint8'),final_labels,49,9)
    final_labels[pix == 1] = result[pix == 1]

    return final_labels

def sgbm(Il,Ir, max_disp):
    # SGBM Parameters -----------------
    window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
     
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)


    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)   
    print('computing disparity...')
    displ = left_matcher.compute(Il, Ir) #.astype(np.float32)/16
    displ = (displ.astype(np.float32))/16

    dispr = right_matcher.compute(Ir, Il).astype(np.float32)/16
    #displ = np.int16(displ)
    #dispr = np.int16(dispr)

    #displ=wls_filter.filter(displ,Il,None,dispr)
    #pdb.set_trace()
    
    #displ = cv2.normalize(src=displ, dst=displ, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    #filteredImg = np.float32(filteredImg)

    return displ

def computeDisp_real(Il,Ir):

    Il_gray=cv2.cvtColor(Il,cv2.COLOR_BGR2GRAY)
    Ir_gray=cv2.cvtColor(Ir,cv2.COLOR_BGR2GRAY)

    #Il_gray=cv2.equalizeHist(Il_gray)
    #Ir_gray=cv2.equalizeHist(Ir_gray)
    #Ir_gray=cv2.imread(Ir_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)

    left_volume, right_volume=get_cost_volume(Il_gray,Ir_gray,15,'pretrained_models/model_epoch2000.ckpt')
    left_volume=left_volume.transpose(1,2,0)
    right_volume=right_volume.transpose(1,2,0)

    '''
    for i in range(15) :
        left_volume[:, :,i] = cv2.bilateralFilter(left_volume[:,: ,i],10,9,16)
        right_volume[:, :,i] = cv2.bilateralFilter(right_volume[:,: ,i],10,9,16)
    '''

    
    #Il=(Il/2).astype(np.uint8)
    #left_volume=cv2.ximgproc.guidedFilter(Il,left_volume,23,9)
    #right_volume=cv2.ximgproc.guidedFilter(Ir,right_volume,23,9)
    #pdb.set_trace()
    h, w, ch = Il.shape
    print ('shape: {},{}'.format(h,w))

    disp = np.zeros((h, w), dtype=np.int32)
    disp = np.argmin(left_volume,axis=2).astype(np.float32)
    disp_r = np.argmin(right_volume,axis=2).astype(np.float32)

    #pdb.set_trace()
    #disp=filler(Il,disp,disp_r, 15)

    disp=cv2.ximgproc.weightedMedianFilter(Il,disp,23,9)
    #disp=cv2.ximgproc.guidedFilter(Il_rect,disp,11,9)
    #disp=cv2.medianBlur(disp,5)
    print ('gg')
    return disp

# You can modify the function interface as you like
def computeDisp(Il_path, Ir_path, max_disp):


    #right_image = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    Il = cv2.imread(Il_path)
    Ir = cv2.imread(Ir_path)
    #Il_rect, Ir_rect, Il_warpmat, Ir_warpmat=rectify(Il,Ir)
    Il_rect, Ir_rect=(Il,Ir)



    Il_gray=cv2.cvtColor(Il_rect,cv2.COLOR_BGR2GRAY)
    Ir_gray=cv2.cvtColor(Ir_rect,cv2.COLOR_BGR2GRAY)

    #Il_gray=cv2.equalizeHist(Il_gray)
    #Ir_gray=cv2.equalizeHist(Ir_gray)
    #Ir_gray=cv2.imread(Ir_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)

    left_volume, right_volume=get_cost_volume(Il_gray,Ir_gray,max_disp,'pretrained_models/model_epoch2000.ckpt')
    left_volume=left_volume.transpose(1,2,0)
    right_volume=right_volume.transpose(1,2,0)

    for i in range(max_disp) :
        left_volume[:, :,i] = cv2.bilateralFilter(left_volume[:,: ,i],10,9,16)
        right_volume[:, :,i] = cv2.bilateralFilter(right_volume[:,: ,i],10,9,16)

    
    #Il=(Il/2).astype(np.uint8)
    left_volume=cv2.ximgproc.guidedFilter(Il_rect,left_volume,23,9)
    right_volume=cv2.ximgproc.guidedFilter(Ir_rect,right_volume,23,9)
    #pdb.set_trace()
    h, w, ch = Il_rect.shape
    print ('shape: {},{}'.format(h,w))

    disp = np.zeros((h, w), dtype=np.int32)
    disp = np.argmin(left_volume,axis=2).astype(np.float32)
    disp_r = np.argmin(right_volume,axis=2).astype(np.float32)

    #pdb.set_trace()
    disp=filler(Il_rect,disp,disp_r, max_disp)

    #disp=cv2.ximgproc.weightedMedianFilter(Il_rect,disp,23,9)
    #disp=cv2.ximgproc.guidedFilter(Il_rect,disp,11,9)
    disp=cv2.medianBlur(disp,5)

    if np.mean(disp)<10:
        disp=computeDisp_real(Il,Ir)

    #disp = cv2.warpPerspective(disp,Il_warpmat,(Il.shape[0],Il.shape[1]),flags=cv2.WARP_INVERSE_MAP)

    '''
    disp=sgbm(Il,Ir,max_disp)
    disp=cv2.ximgproc.weightedMedianFilter(Il_rect,disp,23,9)
    '''

    '''
    Il = cv2.imread(Il_path)
    Ir = cv2.imread(Ir_path)

    #pdb.set_trace()
    #Il=cv2.resize(Il,(Il.shape[0]*3,Il.shape[1]*3))
    #Il=cv2.resize(Il,(int(Ir.shape[0]*1.5),int(Ir.shape[1]*1.5)))
    #Ir=cv2.resize(Ir,(int(Ir.shape[0]*1.5),int(Ir.shape[1]*1.5)))
    Il=cv2.resize(Il,(384,512))
    Ir=cv2.resize(Ir,(384,512))

    Il=Variable(torch.FloatTensor(Il).unsqueeze_(0).permute(0,3,1,2))
    Ir=Variable(torch.FloatTensor(Ir).unsqueeze_(0).permute(0,3,1,2))
    

    m=stackhourglass(64)
    m=nn.DataParallel(m)
    m.cuda()
    m.eval()

    state_dict=torch.load('pretrained_model_KITTI2015.tar')
    #state_dict=torch.load('pretrained_model_KITTI2012.tar')
    #state_dict=torch.load('pretrained_sceneflow.tar')
    m.load_state_dict(state_dict['state_dict'])

    disp=m(Il,Ir).data.cpu().numpy().squeeze()


    #disp=cv2.ximgproc.weightedMedianFilter(guide,disp,17)
    #disp=disp.astype(np.int32).astype(np.float32)
    # TODO: Some magic
    '''

    return disp


def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    '''
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    '''
    tic = time.time()
    disp = computeDisp(args.input_left, args.input_right, 61)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
