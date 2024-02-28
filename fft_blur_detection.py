import os
import cv2
import shutil
import numpy as np

dataset = "blur_objects"
std_dev_limit = 1.3
max_fft_dist = 30
deblur_models = ["mprnet", "restormer", "deeprft"]

def calc_fft_score(image, shift_percent=0.1):
    h, w = image.shape
    (cx, cy) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cy - int(shift_percent * h): cy + int(shift_percent * h), cx - int(shift_percent * w) : cx + int(shift_percent * w)] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean

def calc_blur_score(val, max_fft, max_blur_dist):
    score = (max_fft - val) / max_blur_dist
    if score < 0:
         score = 0
    elif score > 1:
         score = 1
    return score

def move_frames(source_folder, target_folder, image_list):
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    for filename in os.listdir(source_folder):
        if filename in image_list:
            shutil.copy(os.path.join(source_folder, filename), os.path.join(target_folder, filename))
    

def identify_clear_frames(dataset, std_dev_limit):
    fft_magnitude = {}
    folder_dir = os.path.join("data", dataset, "images")
    files = [f for f in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, f))]
    for file in files:
        orig = cv2.imread(os.path.join(folder_dir, file))
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        fft_mean = calc_fft_score(
            gray
        )
        fft_magnitude[file] = fft_mean

    sorted_fft_magnitude = dict(sorted(fft_magnitude.items(), key=lambda item: item[1], reverse=True))
    images = list(sorted_fft_magnitude.keys())
    fft_val = list(sorted_fft_magnitude.values())
    np_fft_val = np.array(fft_val)
    median_fft = np.median(np_fft_val)
    std_dev_fft = np.std(np_fft_val)
    blur_threshold = median_fft + (std_dev_fft * std_dev_limit)
    clear_frames, clear_fft_val, blur_frames, blur_fft_val= [], [], [], []
    for i, value in enumerate(fft_val):
        if value > blur_threshold:
            clear_frames.append(images[i])
            clear_fft_val.append(value)
        else:
            blur_frames.append(images[i])
            blur_fft_val.append(value)

    print(f"Dataset: {dataset}")
    print(f"{len(clear_frames)} clear frames: {clear_frames}")
    move_frames(folder_dir, os.path.join("data", dataset, "clear"), clear_frames)
    move_frames(folder_dir, os.path.join("data", dataset, "blur"), blur_frames)
    print(f"---> Clear and blurry frames copied to seperate folders")

    return clear_frames, clear_fft_val, blur_frames, blur_fft_val

def median_dataset_blur_score(blur_fft_val, max_fft, max_fft_dist, name="dataset"):
    blur_score = [calc_blur_score(value, max_fft, max_fft_dist) for  value in blur_fft_val]
    median_score = np.median(np.array(blur_score))
    print(f"{name.upper()} Median Blur Score: {round(median_score, 2)}")
    return median_score

def calc_deblur_model_blur_scores(dataset, models, max_fft_dist, max_clear_fft,dataset_median_score):
    deblur_model_blur_scores = {}
    for model in models:
        fft_magnitude = {}
        folder_dir = os.path.join("data", dataset, model)
        files = [f for f in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, f))]
        for file in files:
            orig = cv2.imread(os.path.join(folder_dir, file))
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            fft_mean = calc_fft_score(
                gray
            )
            fft_magnitude[file] = fft_mean
            fft_val = list(fft_magnitude.values())
        deblur_model_blur_scores[model] = median_dataset_blur_score(fft_val, max_clear_fft, max_fft_dist, model)
    return deblur_model_blur_scores



clear_frames, clear_fft_val, blur_frames, blur_fft_val = identify_clear_frames(dataset, std_dev_limit)
max_clear_fft = np.max(np.array(clear_fft_val))
dataset_median_score = median_dataset_blur_score(blur_fft_val, max_clear_fft, max_fft_dist)
deblur_model_blur_scores = calc_deblur_model_blur_scores(dataset, deblur_models, max_fft_dist, max_clear_fft,dataset_median_score)





        

