import os
import cv2
import numpy as np
from PIL import Image
import argparse

def BGR2RGB(BGR_img):
    # turning BGR pixel color to RGB
    rgb_image = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    return rgb_image

def sharpenImages(base_dir, new_basedir, directories, scene, imagedir):

    output_dir = new_basedir + scene
    sharp_imgs=[]

    for i,directory in enumerate(directories):
        
        frame = cv2.imread(directory)
        # Create the sharpening kernel
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        # Apply the sharpening kernel to the image using filter2D
        sharpened = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
        # cv2.imshow(sharpened)
        sharpened = BGR2RGB(sharpened)
        sharpened = Image.fromarray(sharpened, 'RGB')
        sharpened.save(output_dir +'/'+imagedir+'/'+directory.split('/')[-1], dpi=(frame.shape[0], frame.shape[1]))

        sharp_imgs.append(i)

    poses_arr = np.load(os.path.join(base_dir+scene+'/', 'poses_bounds.npy'))
    poses_arr = poses_arr[sharp_imgs, :]

    np.save(os.path.join(output_dir, 'poses_bounds.npy'), poses_arr)


def main( base_dir, new_basedir, scenes, imagedir):
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
    sharpenImages(base_dir, new_basedir, directories,scene, imagedir)



if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-b", "--basedir", help="input directory")
    argParser.add_argument("-o","--outputdir", help ="output directory")
    argParser.add_argument('-s','--scenes',help ="scenes to be blur detected. comma seperated")
    argParser.add_argument('-i','--imagedir', help='image directory within the input directory') 

    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.basedir=%s" % args.basedir)
    print("args.outputdir=%s" % args.outputdir)
    print("args.scenes=%s" % args.scenes)
    print("args.imagedir=%s" % args.imagedir)

    scenes = args.scenes.split(',')
    print("args.scenes=%s" % scenes)


    # new_basedir = './deblurnerf_dataset/real_camera_motion_blur/filtered/'
    # base_dir = 'deblurnerf_dataset/real_camera_motion_blur/'
    # scenes = ['blurstair']

    main(args.basedir, args.outputdir, scenes, args.imagedir)