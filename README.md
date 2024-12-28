# Critical-Node-Detection-AI-Research


### Run classic algorithm

```
python3 run_classic_algo.py
```


### Run learning based algorithms

Train MaskGAE :

```
python3 train_mgae.py --dataset terrorist --dataset_path dataset/9-11_terrorists
```

Inference :

```
python3 run_mgae_algo.py --weight_path weights/maskgae_terrorist_epoch500.pth --dataset terrorist --dataset_path dataset/9-11_terrorists
```
