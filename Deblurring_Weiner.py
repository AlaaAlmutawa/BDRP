import numpy as np
import cv2
from skimage import restoration
from scipy.signal import convolve2d as conv2
import argparse
import os
import matplotlib.pyplot as plt

def BGR2RGB(BGR_img):
    # turning BGR pixel color to RGB
    rgb_image = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    return rgb_image
    
def weinerFiltering(base_dir, new_basedir, directories,scene, imagedir, sharpen = False):
  
    output_dir = new_basedir + scene
    deblurred_imgs=[]
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

    for j,directory in enumerate(directories):
        
        frame = cv2.imread(directory)

        b, g, r = cv2.split(frame)

        img_b = np.float32(b)/255.0
        img_g = np.float32(g)/255.0
        img_r = np.float32(r)/255.0

        img_arrays = [img_b, img_g, img_r]
        width = 2
        psf = np.ones((width, width)) / (width * width)
        new_img_arrays = []

        for i in img_arrays:
            image = conv2(i, psf, 'same')
            image += 0.1 * image.std() * np.random.standard_normal(image.shape)

            deconvolved = restoration.unsupervised_wiener(image, psf, clip=True)
            new_img_arrays.append(deconvolved[0])
        
        deconvolved = cv2.merge((new_img_arrays[2],new_img_arrays[1],new_img_arrays[0]))

        if sharpen :
            deconvolved = cv2.filter2D(deconvolved, -1, kernel)

        plt.imsave(output_dir +'/'+imagedir+'/'+directory.split('/')[-1],np.clip(deconvolved, 0,1))
        deblurred_imgs.append(j)

    poses_arr = np.load(os.path.join(base_dir+scene+'/', 'poses_bounds.npy'))
    poses_arr = poses_arr[deblurred_imgs, :]

    np.save(os.path.join(output_dir, 'poses_bounds.npy'), poses_arr)


def main( base_dir, new_basedir, scenes, imagedir, sharpen = False):
  for scene in scenes:
    image_list = os.listdir(base_dir + scene +'/'+imagedir+'/')
    directories = [base_dir + scene +'/'+imagedir+'/'+ x for x in image_list]

    os.makedirs(os.path.dirname(new_basedir+scene +'/'+imagedir+'/'), exist_ok=True)
    # os.makedirs(os.path.dirname(new_basedir+scene+'/sparse/0/'), exist_ok=True)

    # for i in os.listdir(base_dir + scene +'/sparse/0/'):
    #   shutil.copy(base_dir + scene +'/sparse/0/'+i, new_basedir+scene+'/sparse/0/'+i)

    # for i in os.listdir(base_dir + scene ):
    #   if 'hold=' in i:
    #     shutil.copy(base_dir + scene +'/'+i, new_basedir+scene+'/'+i)
    weinerFiltering(base_dir, new_basedir, directories,scene, imagedir, sharpen)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-b", "--basedir", help="input directory")
    argParser.add_argument("-o","--outputdir", help ="output directory")
    argParser.add_argument('-s','--scenes',help ="scenes to be blur detected. comma seperated")
    argParser.add_argument('-i','--imagedir', help='image directory within the input directory') 
    argParser.add_argument('-sh','--sharpen', help='sharpen the output images') 

    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.basedir=%s" % args.basedir)
    print("args.outputdir=%s" % args.outputdir)
    print("args.scenes=%s" % args.scenes)
    print("args.imagedir=%s" % args.imagedir)
    print("args.sharpen=%s" % args.sharpen)

    scenes = args.scenes.split(',')
    print("args.scenes=%s" % scenes)


    # new_basedir = './deblurnerf_dataset/real_camera_motion_blur/filtered/'
    # base_dir = 'deblurnerf_dataset/real_camera_motion_blur/'
    # scenes = ['blurstair']

    main(args.basedir, args.outputdir, scenes, args.imagedir, args.sharpen == 'True')