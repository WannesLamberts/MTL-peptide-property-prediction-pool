  
    
    
# Thesis - Context-Aware Retention Time Prediction for Enhanced Proteomics Analysis
## Introduction
In this thesis, we investigated whether it is feasible and meaningful to add sample context to models for predicting retention time. This is one of the codebases used in the project; however, another portion of this thesis was developed in a separate Git repository, which includes more comprehensive documentation and can be accessed at the following link ([WannesLamberts/Massivekb-Pipeline](https://github.com/WannesLamberts/Massivekb-Pipeline). 
## Setup environment
All required packages can be installed in a conda environment by running:
```bash  
 conda env create -f environment.yml  
 ``` 
 It is also possible to build our apptainer image which was used for all experiments in the thesis.
 ```bash  
 apptainer build env.sif apptainer.def
 ```
 The pretrained model build by Ceder Dens, can be found at [Zenodo](https://doi.org/10.5281/zenodo.11084463).   
 
## Training the model
Trained the context-aware model first the context tokens have to be collect by running:
 ```bash  
 python get_encodings.py --run "lightning_logs/CONFIG=mtl_5foldcv_pretrain_0,TASKS=CCS_iRT,MODE=pretrain,PRETRAIN=none,LR=0.0001940554482365,BS=1024,OPTIM=adam,LOSS=mae,CLIP=False,ACTIVATION=gelu,SCHED=warmup,SIZE=180,NUMLAYERS=9/version_0" --data-file "data/dataset.parquet" --train-i "data/train.csv" --val-i "data/val.csv" --test-i "data/test.csv" --out_file "data/lookup.parquet"
 ```
When this tokens are collect the model can be trained by running:
```bash  
python train.py --config $NAME -p own --checkpoint-id 0 -c --data-file "data/dataset.parquet" --train-i "data/$TRAIN" --val-i "data/val.csv" --test-i "data/test.csv" --hpt-config "hpt/hpt_class.csv" --hpt-id $HPT_ID --vocab-file "data/vocab.p" --lookup "data/lookup.parquet" --bs 2048 --type pool
 ```
 `$name` stands for the name you want the output file to have.
 `$HPT_ID` stands for the hyperparameter configuration id wanted(configurations can be found in `/hpt`.
 `$TRAIN` stands for the csv file containing the data indices for training.
## Parameter tuning
The parameter tuning setup can be run by: 
```bash
python param_tune.py --checkpoint-id 0 --data-file "data/dataset.parquet" --train-i "data/train.csv" --val-i "data/val.csv" --vocab-file "data/vocab.p" --lookup "data/lookup.parquet" --bs 2048 --type pool --epochs 10 --amount 6 --optuna 42
```
## Predict
Running prediction with the trained model can be done by
```bash
python predict.py --run $DIR --all_data_file "data/$DATASET/all_data.csv" --predict_i "data/$DATASET/test.csv" --lookup_file "data/$DATASET/lookup.parquet"
```
where `$DIR` stands for the directory the model checkpoint is located in

## Code structure  
All our analysis and results can be found in the `/notebooks` directory.
In `/slurms_scripts` All executed scripts can be found that we used during this thesis for experiments.
In `/src` All main components can be found.
in `hpt/hpt_class` the main hyperparameter configurations can be found.
`Get_encodings.py` has the implementation for building the context tokens.
`param_tuning.py` has the implementation for the parameter tuning functionality.
`predict.py` has the implementation for predicting using a trained model.
`train.py` has the implementation for training a model.

## Acknowledgements
We build this codebase on the original code by Ceder Dens for the base model.