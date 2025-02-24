# Comparison of various Deblurring Algorithms for 3D Reconstruction using NeRF
## Setup Environments: 
## Ruche Cluster: 

### Start GPU runtime

```
srun --nodes=1 --time=18:00:00 -p gpua100 --gres=gpu:1 --mem=50G --export=none --pty /bin/bash
```

```
## to check the gpu job list
squeue
```

### Initialize Conda Environment

```
module load anaconda3/2022.10/gcc-11.2.0
conda create env --name <env-name> --file env.yml
source activate deblur_nerf
```

### Reload Conda Environment

```
module load anaconda3/2022.10/gcc-11.2.0
source activate deblur_nerf
```


## Methodology: 

- Litreture Review and Related Work
- Research Question definition
- Metrics deifnition
- Implementation
- Results
- Discussion and Analysis

#### Goal: 
### Research the possibility of decoupling the deblurring process from the 3d reconstruction of noval views. 
### Given abnormal input with motion blur, is it possible to decouple the deblurring step from the 3d reconstruction of the novel views and get acceptable results? 
## Experiments: 
#### Techniques explored
![alt text](https://github.com/AlaaAlmutawa/BDRP/blob/main/diagrams/Taxonomy-Techniques.png)
#### Metrics:
![alt text](https://github.com/AlaaAlmutawa/BDRP/blob/main/diagrams/Taxonomy-Metrics.png)
#### Dataset: 
![alt text](https://github.com/AlaaAlmutawa/BDRP/blob/main/diagrams/dataset_overview.png)
### Proposed pipeline
![alt text](https://github.com/AlaaAlmutawa/BDRP/blob/main/diagrams/Proposed_pipeline.png)
### Execution of the pipeline
#### Pre-requistes: 
1. DeepRFT
   * Clone [DeepRFT repository](https://github.com/invokerer/deeprft)
2. NeRF
   * Clone [Deblur-NeRF repository](https://github.com/limacv/Deblur-NeRF)
   * We have utilized this repository as it can be used to run both deblur-nerf and vanilla nerf. As per the authors' instruction, vanilla nerf can be ran when setting the configuration of kerneltype=none.
#### Steps 
1. Load the dataset
2. Extract clear frames
   - Note that expected directory of stored images is as follows >> data/dataset_name/images
   - Result of this, will be two new folders populated in the dataset folder, one for blur frames and one for clear frames
```bash
python fft_blur_detection.py --dataset dataset_name --purpose clear_frames
```
3. Deblur images

Currently, we are limited by the assumption that we apriori whether the blur in the images is synthetic or real. (automated detection of the type of blur is left for future work).

If the blur is Synthetic, we execute DeepRFT model with Go-Pro plus pretrained weights 
```bash
python test.py --weights pretrained/DeepRFT_PLUS/model_GoPro.pth --input_dir data/blurwine/blur --output_dir data/blurwine/plus_gopro_results --win_size 256 --num_res 20
```
If the blur is Real, we execute DeepRFT model with RealBlur_j pretrained weights 
```bash
python test.py --weights pretrained/DeepRFT_PLUS/model_RealBlurJ.pth --input_dir data/test/blur --output_dir data/test/deblur_results --win_size 256 --num_res 20
```
4. Combine the clear frames extracted in step 2 with the deblurred frames
5. Execute vanilla NeRF
```bash
python3 run_nerf.py --config configs/<config_path> --kernel_type=none
```
#### Retrieving blur score 
a. For original dataset
```bash
python fft_blur_detection.py --dataset dataset_name --purpose dataset_blur_score
```
b. For deblur results
   - Note: Expected directory structure for each model results >> data/dataset_name/model
```bash
python fft_blur_detection.py --dataset dataset_name --purpose deblur_result_blur_score --deblur_models model1,model2,model3
```
#### Dataset and experiments results 
* blur_objects dataset and its corresponding poses can be accessed through [link](https://drive.google.com/drive/folders/1TfuY0mkoK7vQ0UoM6S_E1ibNGXxfmoM8)
* all experiments results can be accessed through [link](https://drive.google.com/drive/folders/13oSU2SKoaJWBSJYInp2-am3PX5zuP33p?usp=sharing)







