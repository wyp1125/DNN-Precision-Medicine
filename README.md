# Python-Deep-Learning-Toolset
Python programs for applying deep neural network models to new datasets. New datasets should be divided into a training and a testing CSV file. Data should be organized in NumPy DataFrame format, with first row being variable/feature names and last column being label. Categorical values should show characters rahter than numbers.

1) divide_train_test.py.

This program divides a dataset into train and testing sets.

2) make_test_data_cat.py.

This program makes a simulated dataset for DNN application programms. The last several (no more than 10) columns before 'label' contain categorical values. If no categorical values are simulated, please supply the last argument with 0.

For instance,  "python make_test_data_cat.py sample1.train.csv 3 200 20 4" to simulate a training dataset with 3 classes, 200 samples, 20 features (last 4 are categorical); "python make_test_data_cat.py sample2.test.csv 3 50 10 0" to simulate a testing dataset with 3 classes, 50 samples, 10 features (all are real values).

3) dnn_model_pandas_num.py.

DNN model to process new datasets of all real values.

4) dnn_model_pandas_cat.py.

DNN model to process new datasets containing categorical columns
