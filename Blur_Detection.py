## Reference 
## https://www.kaggle.com/code/eladhaziza/perform-blur-detection-with-opencv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import argparse

def variance_of_laplacian(img2):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def BGR2RGB(BGR_img):
    # turning BGR pixel color to RGB
    rgb_image = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    return rgb_image


def blurrinesDetection(base_dir, new_basedir, directories, scene, threshold):

    output_dir = new_basedir + scene
    columns = 3
    rows = len(directories)//2
    fig=plt.figure(figsize=(5*columns, 4*rows))
    sharp_imgs=[]
    for i,directory in enumerate(directories):
        fig.add_subplot(rows, columns, i+1)
        img = cv2.imread(directory)
        text = "Not Blurry"
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry
        fm = variance_of_laplacian(img)
        if fm < threshold:
            text = "Blurry"
        else:
          shutil.copy(directory, output_dir +'/images/'+directory.split('/')[-1])
          sharp_imgs.append(i)
        rgb_img = BGR2RGB(img)
        cv2.putText(rgb_img, "{}: {:.2f}".format(text, fm), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        plt.imshow(rgb_img)

    poses_arr = np.load(os.path.join(base_dir+scene+'/', 'poses_bounds.npy'))
    poses_arr = poses_arr[sharp_imgs, :]

    np.save(os.path.join(output_dir, 'poses_bounds.npy'), poses_arr)
    plt.show()



def main( base_dir, new_basedir, scenes):
    for scene in scenes:
        image_list = os.listdir(base_dir + scene +'/images/')
        directories = [base_dir + scene + '/images/' + x for x in image_list]

        os.makedirs(os.path.dirname(new_basedir+scene+'/images/'), exist_ok=True)
        # os.makedirs(os.path.dirname(new_basedir+scene+'/sparse/0/'), exist_ok=True)

        # for i in os.listdir(base_dir + scene +'/sparse/0/'):
        #     shutil.copy(base_dir + scene +'/sparse/0/'+i, new_basedir+scene+'/sparse/0/'+i)

        # for i in os.listdir(base_dir + scene ):
        #     if 'hold=' in i:
        #         shutil.copy(base_dir + scene +'/'+i, new_basedir+scene+'/'+i)
        blurrinesDetection(base_dir, new_basedir, directories, scene, 200)


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-b", "--basedir", help="input directory")
    argParser.add_argument("-o","--outputdir", help ="output directory")
    argParser.add_argument('-s','--scenes',help ="scenes to be blur detected. comma seperated")    

    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.basedir=%s" % args.basedir)
    print("args.outputdir=%s" % args.outputdir)
    print("args.scenes=%s" % args.scenes)

    scenes = args.scenes.split(',')
    print("args.scenes=%s" % scenes)


    # new_basedir = './deblurnerf_dataset/real_camera_motion_blur/filtered/'
    # base_dir = 'deblurnerf_dataset/real_camera_motion_blur/'
    # scenes = ['blurstair']

    main(args.basedir, args.outputdir, scenes)