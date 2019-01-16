from util import readPFM, writePFM, cal_avgerr
from main import computeDisp
from os.path import join
import pdb
import argparse
import time
import numpy as np
import cv2
from image_rectification import rectify

def main( args ):
        if args.disp:
                for i in range(10):
                        if (args.mode=='s'):
                            fnameL = './data/Synthetic/TL{}.png'.format(i)
                            fnameR = './data/Synthetic/TR{}.png'.format(i)

                        if (args.mode=='r'):
                            fnameL = './data/Real/TL{}.bmp'.format(i)
                            fnameR = './data/Real/TR{}.bmp'.format(i)

                        fnameO = join( args.disp_dir, 'TL{}.pfm'.format(i) )
                        print( 'Compute disparity for {}'.format(fnameO) )
                        img_left  = cv2.imread( fnameL )
                        img_right = cv2.imread( fnameR )

                        Il_rect, Ir_rect, Il_warpmat,Ir_warpmat = rectify(img_left,img_right)

                        tic = time.time()
                        #disp = computeDisp(img_left, img_right)
                        disp = computeDisp(fnameL, fnameR , args.max_disp)
                        toc = time.time()
                        writePFM(fnameO, disp)
                        print('Elapsed time: %f sec.' % (toc - tic))
                        
                        #visualize
                        fnamepng = join( args.disp_dir, 'TL{}.png'.format(i) )
                        Ilrectpng = join( args.disp_dir, 'TL{}_rect.png'.format(i) )
                        Irrectpng = join( args.disp_dir, 'TR{}_rect.png'.format(i) )
                        #pdb.set_trace()
                        disp=(disp/np.max(disp)*255).astype(np.uint8)
                        cv2.imwrite(fnamepng, disp)
                        cv2.imwrite(Ilrectpng, Il_rect)
                        cv2.imwrite(Irrectpng, Ir_rect)
        if args.eval:
                err = np.zeros(10)
                for i in range(10):
                        fnameMy = join( args.disp_dir, 'TL{}.pfm'.format(i) )
                        fnameGT = './data/Synthetic/TLD{}.pfm'.format(i)
                        My = readPFM(fnameMy).reshape(-1)
                        GT = readPFM(fnameGT).reshape(-1)
                        err[i] = cal_avgerr(GT, My)
                # write csv
                fp = open( args.output, 'w' )
                fp.write( 'id,error\n' )
                for i in range(10):
                        fp.write('TL{},{}\n'.format(i, err[i]))
                fp.write('mean,{}'.format(np.mean(err)))
                print (np.mean(err))

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Disparity Estimation')
        parser.add_argument('-d', '--disp', action='store_true', help='write the disp file in pfm')
        parser.add_argument('-e', '--eval', action='store_true', help='evaluate the error')
        parser.add_argument('-m', '--mode', help='synthetic or real data')
        parser.add_argument('-md', '--max_disp', default=60 , type=int, help='max disparity')
        parser.add_argument('--disp_dir', default='disp_map', type=str, help='disparity dir')
        parser.add_argument('--output', default='result.csv', type=str, help='output file')

        args = parser.parse_args()
        main( args )
