# BDRP Project 
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
```bash
command
```
3. Deblur images

Currently, we are limited by the assumption that we apriori whether the images the blur in the images is synthetic or real. (automated detection of the type of blur is left for future work).

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
command
```
#### Retrieving blur score 

```bash
command
```
#### Dataset and experiments results 
* blur_objects dataset and its corresponding poses can be accessed through [link](https://drive.google.com/drive/folders/1TfuY0mkoK7vQ0UoM6S_E1ibNGXxfmoM8)
* all experiments results can be accessed through [link](https://drive.google.com/drive/folders/1L_MY4IAPPEpqEMyJrcCFgnFpDCdUlZz2)







