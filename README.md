# Boundary-Aware Cascade Networks for Temporal Action Segmentation
> Pytorch implementation of Boundary-Aware Cascade Networks for Temporal Action Segmentation (ECCV 2020).
>
> Two proposed novel components: (1) stage cascade for boosting segmentation accuracy for hard frames (e.g., near action boundaries); and (2) local barrier pooling which uses boundary information for more smooth prediction and less over-segmentation errors.

### Visualization video
To be done...

### Updates

Aug, 2020 - We uploaded the code for 50salads, Breakfast and GTEA datasets, and the corresponding models for inference.

### Dependencies

1. Python 3.5
2. PyTorch 0.4.1
3. tensorboard and tensorboardX

### Download Datasets 

* Download the [data](https://zenodo.org/record/3625992#.Xiv9jGhKhPY) provided by [MS-TCN](https://github.com/yabufarha/ms-tcn),  which contains the I3D (w/o fine-tune) features and the ground truth labels for 3 datasets. (~30GB)
* Extract it so that you have the `data` folder in the same directory as `main.py`.

### Training and Testing of BCN
* All the following `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other two datasets. 
* For each dataset, we need to train a model of all the splits, and report the average performance on splits as the final result. We kindly remind that although the performance behavior on different splits varies a lot, we should still select the models of the same epoch for all splits in a specific dataset.

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
The predicted barriers (selected from boundary confidence scores) is saved in .csv file.
```
python bgm.py --action test --dataset DS --split SP --resolution resized
```

#### 4. Training our Stage Cascade  and BGM jointly
We will freeze the parameters of BGM for the first several epochs and jointly optimize two modules until convergence. The performance displayed on screen and tensorboardX is only BCN without post-processing, which is **NOT** our final result.
```
python main.py --action train --dataset DS --split SP
```
We also provide trained full and resized BGM model in [this mega link](https://mega.nz/file/GGoz3JRA#FsTyOATlWJ3oh7-fE7cmPw4GUsHpg_1Oz9BxBtrhLSQ). Extract the zip file `bcn_model.zip` and put the `best_models` folder in the same directory as `main.py`.

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
Limited by the size of temporal action segmentation datasets, the convergence of training procedure is not satisfied, where the performance difference between adjacent epochs may be larger than 1 percent in all metrics  (especially for GTEA dataset). My empirical solution is evaluating all the saved models and selecting the epoch of best average performance. 

Due to the random initialization, we think that the training result is good if the performance gap for most of metrics between your training result and the provided model is less than

* 0.3% in Breakfast dataset
* 0.8% in 50salads dataset
* 2% in GTEA dataset

It is also common if your result is better than the performance reported in our paper.



### Citation:

If you find our code useful, please cite our paper.

### Contact

For any question, please raise an issue or contact

```
Zhenzhi Wang: zhenzhiwang@outlook.com
```

