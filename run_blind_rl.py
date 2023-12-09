import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve, convolve
from scipy.ndimage import gaussian_filter
from skimage import color, data, restoration
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy.signal import fftconvolve
import os
import argparse


def create_gaussian_psf(size, sigma):
    """
    Creates a Gaussian Point Spread Function (PSF).

    :param size: The size of the PSF array (it will be a square).
    :param sigma: The standard deviation of the Gaussian.
    :return: A 2D numpy array representing the PSF.
    """
    psf = np.zeros((size, size))
    psf[size // 2, size // 2] = 1  # Create a single peak at the center
    psf = gaussian_filter(psf, sigma=sigma)
    return psf

def richardson_lucy_blind_3_color(image, psf, num_iter=1):
    # Check if image is grayscale or color
    if image.ndim == 2:
        # Grayscale image
        return richardson_lucy_blind_3(image, psf, num_iter)
    elif image.ndim == 3:
        # Color image with multiple channels
        channels = image.shape[2]
        im_deconv = np.zeros_like(image, dtype='float')

        for ch in range(channels):
            # Process each channel separately
            im_deconv[:, :, ch] = richardson_lucy_blind_3(image[:, :, ch], psf, num_iter)

        return im_deconv
    else:
        raise ValueError("Image must be either 2D (grayscale) or 3D (color)")

def richardson_lucy_blind_3(channel, psf, num_iter=1):
    im_deconv = np.full(channel.shape, 0.1, dtype='float')    # init output
    for i in range(num_iter):
        psf_mirror = np.flip(psf)
        conv = fftconvolve(im_deconv, psf, mode='same')
        relative_blur = channel / conv
        im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')
        im_deconv_mirror = np.flip(im_deconv)
        center_x, center_y = 300, 200 ##insert dimensions
        slice_5x5 = (slice(center_x - 3,center_x + 2),slice(center_y - 3,center_y + 2))
        # center_slice = (slice(253, 258), slice(253, 258))  # 5x5 slice
        small_im_deconv_mirror = im_deconv_mirror[slice_5x5]
        conv_result = fftconvolve(relative_blur[slice_5x5], small_im_deconv_mirror, mode='same')
        # conv_result = fftconvolve(relative_blur, im_deconv_mirror, mode='same')
        if conv_result.shape != psf.shape:
            raise ValueError(f"Convolution result shape {conv_result.shape} does not match psf shape {psf.shape}")
        psf *= conv_result
    return im_deconv


# WORK_DRIVE = '/content/drive'
# FOLDER = '/MyDrive/BDRP/BDRP'
# WORK_AREA = WORK_DRIVE + FOLDER

# drive.mount(WORK_DRIVE)
# # os.chdir(WORK_AREA+'/Deblur-NeRF')


def apply_blind_rl_on_dataset(source_dir,height,width,dest_dir):
  files = os.listdir(source_dir)
  for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img1_path = source_dir+'/' + file
        image_index = os.path.splitext(os.path.basename(img1_path))[0]
        # img path as inputs.
        image = Image.open(img1_path)
        image_np = np.array(image)
        psf = create_gaussian_psf(5, 2)  # Adjust the sigma as needed
        deblur = richardson_lucy_blind_3_color(image_np, psf, 15)
        image_data_deblurred = deblur.reshape(height, width, 3)
        image_data_deblurred = image_data_deblurred.astype(np.uint8)

        im = Image.fromarray(image_data_deblurred)
        # os.chdir(dest_dir)
        im.save(dest_dir+'/'+file)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-source", "--sourcedir", help="source images directory")
    argParser.add_argument("-dest","--destdir", help ="destination images directory")
    argParser.add_argument('-width','--width',help ="width of images")
    argParser.add_argument('-height','--height',help='height of images') 
    args = argParser.parse_args()
    # print("args=%s" % args)

    if len(vars(args)) ==4: 
        apply_blind_rl_on_dataset(args.sourcedir,int(args.height),int(args.width),args.destdir)
    else:
        print('incorrect argument list')