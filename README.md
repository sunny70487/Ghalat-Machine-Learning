<h1>GML - Ghalat Machine Learning!</h1>

<b>Tired of training multiple models and then picking the best among them? No worries now! GML is here for you!</b>
<br>
<br>
GML is an automatic machine learning library in python built on top of Scikit-Learn,Keras,XGBoost,LightGBM and Catboost. with this library, you can train your data on multiple machine learning algorithms and a neural network! not only training but scaling the data for normal distribution and after scaling and training, testing the data on validation data (don't worry you don't need to provide validation data. we will extract it from your data). after testing models on validation data, they will be ranked accordingly and you will see which one performs better than other. the first ranked model will be returned (untrained, so you can train it yourself and check results). You already got some models? no problem! pass them to us to make them compete with our models and let see who wins ;-)<br>
  <br>
In future updates many other things will also be automated like hyper parameter training, multiple neural networks, other machine learning algorithms and many more cool things!
<br>
<br>
<h2>Install it: </h2> <br>
> <b>pip install GML</b>
<br>
<br>
Demo is also provided! <b> <br>
  for Classification tasks: <br>
  for Regression tasks: <br></b>
<br>
<h2>Function description:</h2><br>
These parameters are common in both GMLRegressor and GMLClassifier<br>
    - X 
      Data column excluding the target column. it can either be a pandas dataframe or a numpy array. but please make sure your data doesn't contains missing data or non-numeric data. (clean it before passing)
    - y 
      The targeted column
  Below parameters are optional.
    - metric
      metric on which you want to test your model. by default, it is mean-squared-error for regression and accuracy score for classification
    - test_Size 
      size to split your test data, by default = 0.3 (70% training 30% testing)
    - folds (only in GMLClassifier)
      Data will also be validated using KFolds. pass number of folds. by default folds = 5
    - shuffle
      Shuffle the data when spliting for validation. by default = True
    - scaler
      for Scaler pass:  
        'SS' for StandardScalar
        'MM' for MinMaxScalar
        'log' for Log scalar
         None for not scaling
      by default: StandardScalar
    - models
      You got your own models to make them compete with our models? pass them in a list here. default = None
    - neural_net
      Want to train on Neural Networks? Pass 'Yes', default = 'No'
    - epochs
      for neural networks, by default = 10 
    - verbose
      for neural networks, by default = True
