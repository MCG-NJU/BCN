# Boundary-Aware Cascade Networks for Temporal Action Segmentation (ECCV'20)
> Pytorch implementation of Boundary-Aware Cascade Networks for Temporal Action Segmentation (ECCV 2020).
>
> Two proposed novel methods: (1) stage cascade for boosting segmentation accuracy for hard frames (e.g., near action boundaries); and (2) local barrier pooling which uses boundary information for more smooth prediction and less over-segmentation errors. Our unified framework BCN with these two complementary components outperforms previous SOTA by large margin.
> 
> Please visit our presentation video and slides in ECCV website. Our slides is also available in `./demo/ECCV20-BCN.pdf`.

### Visualization video
To be done...

### Updates

Aug, 2020 - We uploaded the code for 50salads, Breakfast and GTEA datasets, and the corresponding models for inference.

### Dependencies

1. Python 3.5
2. PyTorch 0.4.1
3. tensorboard and tensorboardX

### Download Datasets 

* Download the [data](https://zenodo.org/record/3625992#.Xiv9jGhKhPY) provided by [MS-TCN](https://github.com/yabufarha/ms-tcn),  which contains the I3D features (w/o fine-tune) and the ground truth labels for 3 datasets. (~30GB)
* Extract it so that you have the `data` folder in the same directory as `main.py`.

### Training and Testing of BCN
* All the following `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other two datasets. 
* For each dataset, we need to train a model of all the splits, and report the average performance on splits as the final result.

#### 1. Training full-resolution barrier generation module

The best model of full BGM is for the initialization of BGM in joint-training. By our experiment, only full-resolution BGM can be used for joint-training.
```
python bgm.py --action train --dataset DS --split SP --resolution full
```

#### 2. Training resized-resolution barrier generation module

The predicted boundary by best model of resized BGM is for the post-processing. The quality of boundary predicted by resized BGM is relatively better.
```
python bgm.py --action train --dataset DS --split SP --resolution resized
```

We also provide trained full and resized BGM model in [this mega link](https://mega.nz/file/CChHnLTY#Sr4pRdyAN2PMhTaQhbKfili5mFy9-ICXW9d-kyS-H4o). Extract the zip file `bgm_model.zip` and put the `best_bgm_models` folder in the same directory as `main.py`.

#### 3. Testing resized-resolution barrier generation module
The predicted barriers (selected from boundary confidence scores) is saved in .csv file. Note that we don't use resized LBP for post-processing for `gtea` dataset due to bad barrier quality caused by very small dataset size. But the performance gain in `50salads` and `breakfast` from resized LBP is quite satisfied.
```
python bgm.py --action test --dataset DS --split SP --resolution resized
```

#### 4. Training our Stage Cascade  and BGM jointly
We will freeze the parameters of BGM for the first several epochs and jointly optimize two modules until convergence. Here we only use frame-wise classification loss and optimize BGM by backward gradients. The evaluation both on training and testing set will show on screen during training procedure.
```
python main.py --action train --dataset DS --split SP
```
We also provide trained BCN model in [this mega link](https://mega.nz/file/GGoz3JRA#FsTyOATlWJ3oh7-fE7cmPw4GUsHpg_1Oz9BxBtrhLSQ). Extract the zip file `bcn_model.zip` and put the `best_models` folder in the same directory as `main.py`.

#### 5. Testing our BCN

We directly provide the evaluation result of our final result after running
```
python main.py --action test --dataset DS --split SP
```
The final performance is made by the combination of Stage Cascade, 1 full-LBP and several times of resized-LBP as post-processing.

If you use our provided model, just run step 3) and 5) and you will get the evaluation. To use provided model, keep `use_saved_model=True` in both `main.py` and `bgm.py`.

#### 6.  Evaluation
You can still evaluate again the performance of result predicted in step 5) by running `python eval.py --dataset DS --split SP`, but it is not necessary. Our evaluation code follows [MS-TCN](https://github.com/yabufarha/ms-tcn).


#### About the performance
Limited by the size of temporal action segmentation datasets, the fluctuating training procedure makes the performance difference between adjacent epochs maybe larger than 1 percent in all metrics  (especially for `GTEA` dataset) in many action segmentation methods including ours. My empirical solution is evaluating all the saved models and selecting the epoch of best average performance (over splits). All the metrics are important and their behaviours are similar, so I tend to choose better F1-score because segmentation accuracy in our method is very stable.

Due to the random initialization, we think that the training result is good if the performance gap for most of metrics between your training result and the provided model is less than

* 0.5% in Breakfast dataset
* 1% in 50salads dataset
* 2% in GTEA dataset

Actually the performance reported in our paper is lower than our provided model for better reproducibility because of unstable training process. It is common if your training result is better than ours.


### Citation

If you find our code useful, please cite our paper.

### Contact

For any question, please raise an issue or contact

```
Zhenzhi Wang: zhenzhiwang@outlook.com
```
### Acknowledgement

We appreciate [MS-TCN](https://github.com/yabufarha/ms-tcn) for extracted I3D feature, and codes for backbone model and evaluation. 
