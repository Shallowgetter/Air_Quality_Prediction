# Air_Quality_Prediction

 ## This repository contains complete code and data for program "Air_Quality_Prediction". ##

 ### Data Resources
 We collected data online, and transformed those `CSV` files to `json` form. You can find these data in `111_json` and `111_test`. The `on_process` folder stores data processed before training.

 ### Tips for Training
 There still leaves pity that we haven't provide a whole automatic train-test process for this program. If you are willing to training this model, you need to correct address in `dataset.py`. 
 + In function `load_train_data()` correct `data_directory`;
 + In function `run_test_predictions(model, scaler, selected_sites)` correct `test_directory`;
 + in function `save_preprocessed_data()` correct `data_directory` and `output_path`, also the `os.makedirs(r'D:\Air_Quality_Prediction\data', exist_ok=True)` needs correcting.

 ### Expected Performance
 We have gone through whole process of this program and got the results locally. These results include two different methods used in prediction. Some of them are as follows:

![Transformer_01](https://github.com/Shallowgetter/Air_Quality_Prediction/blob/main/pic/accuracy.png)![accuracy](https://github.com/Shallowgetter/Air_Quality_Prediction/blob/main/pic/results/Transformer_01.png)

