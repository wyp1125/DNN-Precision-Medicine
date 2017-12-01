# Python-DNN-DataFrame
Python programs for applying deep neural network models to new datasets in the form of pandas data frames. New datasets should be divided into a training and a testing CSV file. Data should be organized in pandas DataFrame format, with first row being variable/feature names and last column being label. Categorical values should show characters rahter than numbers.

1) divide_train_test.py.

This program divides a dataset into train and testing sets.

2) make_test_data_cat.py.

This program makes a simple simulated dataset for testing the execution of the DNN programms. The last several (no more than 10) columns before 'label' contain categorical values. If no categorical values are simulated, please supply the last argument with 0.

For instance,  "python make_test_data_cat.py sample1.train.csv 3 200 20 4" to simulate a training dataset with 3 classes, 200 samples, 20 features (last 4 are categorical); "python make_test_data_cat.py sample2.test.csv 3 50 10 0" to simulate a testing dataset with 3 classes, 50 samples, 10 features (all are real values).

3) dnn_model_pandas_num.py.

Build DNN models on new datasets of all real values.

4) dnn_model_pandas_cat.py.

Build DNN models on new datasets containing categorical columns.

5) simulate_pm.py

Generate the simulated precision medicine dataset.

6) pm.train.csv, pm.test.csv

Sample precision medicine modeling data. Use the command "python3 dnn_model_pandas_cat.py pm.train.csv pm.test.csv smoking,alcohol,excercise,substance,depression 4 600,600,600 800" to run.
