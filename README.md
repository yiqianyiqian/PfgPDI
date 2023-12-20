## About PfgPDI
PfgPDI: Pocket Feature-Enabled Graph Neural Network for Protein-Drug Interaction Prediction
PfgPDI is a deep learning architecture, which predicts the binding affinity between ligands and proteins.  

The benchmark dataset can be found in `./data/`. 
Data preprocessing can be referred to `./prepare/`. 
The PfgPDI model is available in `./src/`. 
And the result will be generated in `./runs/`. See our paper for more details.

### Requirements:
- python 3.7
- cudatoolkit 10.1.243
- cudnn 7.6.0
- pytorch 1.4.0
- numpy 1.16.4
- scikit-learn 0.21.2
- pandas 0.24.2
- tensorboard 2.0.0
- scipy 1.3.0
- numba 0.44.1
- tqdm 4.32.1

The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment_gpu.yml
conda activate PfgPDI_env
```

Then, install the apex in the PfgPDI_env environment:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```
Since the codes for apex package in the above website have been updated, you can also install it using the uploaded package. The apex.tar can be found in `./src/`.  


### Training & Evaluation

to train your own model
```bash
cd ./src/
python main.py
```
to see the result
```bash
tensorboard ../runs/PfgPDI_<datetime>_<seed>/
```

### Citation
 Yiqian Zhang,Changjian Zhou. PfgPDI: Pocket Feature-Enabled Graph Neural Network for Protein-Drug Interaction Prediction.

### contact
Changjian Zhou: zhouchangjian@neau.edu.cn
Yiqian Zhang: 815978940@qq.com