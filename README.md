# epl_dataset
Prediction datasets for english premier league

## Why ?
- Everyone should be able to pip install library and build machine learning models on this dataset in few minutes without worrying about data quality and baselines. 
- Library should be bug free, easy to install/use, provide comparison with benchmark models. 
- If you have a fancy algorithm for time series prediction, you can measure performance of your model against baselines. 
- Use this dataset to power your FPL systems. 


## How ?
- Build Wrapper around varav's fpl data and builds a pytorch contextual prediction dataset
- Build prediction dataset.
    - `trainset, testset = get_epl_dataset(feature_names, prediction_names, window_size, batch_size, years)`
    - Dataset has input batches of shape (N, InD, L) and output batches of shape (N, OutD, L)
- Show demo on colab
    - Use epl_dataset.py to benchmark some basic models 
    - search over feature space in another notebook
    - Add visuals over predictions 

