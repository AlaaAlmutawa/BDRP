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

## Goal: 
### Research the possibility of decoupling the deblurring process from the 3d reconstruction of noval views. 
### Given abnormal input with motion blur, is it possible to decouple the deblurring step from the 3d reconstruction of the novel views and get acceptable results? 
#### Plan: 
#### Techniques explored
![alt text](https://github.com/AlaaAlmutawa/BDRP/blob/main/diagrams/Taxonomy-Techniques.png)
#### Metrics:
![alt text](https://github.com/AlaaAlmutawa/BDRP/blob/main/diagrams/Taxonomy-Metrics.png)
#### Dataset: 






