# from google.colab import drive
import numpy as np
from urllib.request import pathname2url
import os
import pandas as pd
import os
import time
import pyiqa
import cv2
import os
import numpy as np
import argparse

# WORK_DRIVE = '/content/drive'
# FOLDER = '/MyDrive/BDRP/BDRP'
# colab = True 

# WORK_AREA = WORK_DRIVE + FOLDER
# if (colab):
#   drive.mount(WORK_DRIVE)

def get_matric(type,metric,iqa_metric,dir):
  files = os.listdir(dir)
  metrics = []
  for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img1_path = dir+'/' + file
        image_index = os.path.splitext(os.path.basename(img1_path))[0]
        # img path as inputs.
        score_fr = iqa_metric(img1_path)

        metrics.append({"image": image_index,
                        "metric_name": metric,
                        f'{type}':score_fr.item()
                        }
                       )
        # print(image_index)
  return pd.DataFrame(metrics)

def record_metrics(dir_gt,dir_blur,dir_deblur,deblur_tech,metric):
  iqa_metric = pyiqa.create_metric(metric)
  dataframe_gt = get_matric('gt',metric,iqa_metric,dir_gt)
  dataframe_blur = get_matric('blurred',metric,iqa_metric,dir_blur)
  dataframe_deblur = get_matric(f'{deblur_tech}_deblurred',metric,iqa_metric,dir_deblur)
  df3=pd.merge(dataframe_gt,dataframe_blur, on=['image','metric_name'])
  df4 = pd.merge(df3,dataframe_deblur, on=['image','metric_name'])
  return df4

def calculate_psnr(dir_gt, dir_deblurred):
    psnr_values = []
    filenames_gt = os.listdir(dir_gt)
    filenames_deblurred = os.listdir(dir_deblurred)

    for filename in filenames_gt:
        if filename in filenames_deblurred:
            path_gt = os.path.join(dir_gt, filename)
            path_deblurred = os.path.join(dir_deblurred, filename)

            gt_image = cv2.imread(path_gt)
            deblurred_image = cv2.imread(path_deblurred)

            if gt_image is not None and deblurred_image is not None:
                psnr = cv2.PSNR(gt_image, deblurred_image)
                image_index = os.path.splitext(os.path.basename(filename))[0]
                psnr_values.append({"image":image_index, 
                                    "psnr":psnr})

    return pd.DataFrame(psnr_values)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-gt", "--gtdir", help="ground truth images directory")
    argParser.add_argument("-db","--deblurdir", help ="deblurred images directory")
    argParser.add_argument('-blurdir','--blurdir',help ="blurred images directory")
    argParser.add_argument('-psnr','--psnr',  action="store_true" , help='calculate psnr only') 
    argParser.add_argument('-niqe','--niqe', action="store_true"  , help='calculate niqe') 
    argParser.add_argument('-brisque','--brisque', action="store_true" , help='calculate brisque')
    argParser.add_argument('-output','--outputdir', help='output directory')
    argParser.add_argument('-method','--method', help='method used to deblur the image')


    args = argParser.parse_args()
    # print("args=%s" % args)

    if args.brisque: 
        df = record_metrics(args.gtdir,args.blurdir,args.deblurdir,args.method,'brisque')
        # csv_path = WORK_AREA+'/blind_rl/'
        # os.chdir(csv_path)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(args.outputdir+f'/results_brisque_{args.method}_{timestr}.csv')
    
    if args.niqe: 
        df = record_metrics(args.gtdir,args.blurdir,args.deblurdir,args.method,'niqe')
        # csv_path = WORK_AREA+'/blind_rl/'
        # os.chdir(csv_path)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(args.outputdir+f'/results_niqe_{args.method}_{timestr}.csv')
    
    if args.psnr: 
        print('test')
        df = calculate_psnr(args.gtdir,args.blurdir)        
        # csv_path = WORK_AREA+'/blind_rl/'
        # os.chdir(csv_path)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(args.outputdir+f'/results_psnr_{args.method}_{timestr}.csv')
