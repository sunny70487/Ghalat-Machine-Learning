<h1>GML - Ghalat Machine Learning! <img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence9-512.png" alt="Brain+Machine" height="38" width="38"> </img> <img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence15-512.png" alt="Adding AI" height="38" width="38"> </img> <img src="https://cdn1.iconfinder.com/data/icons/science-technology-outline/91/Science__Technology_35-512.png" alt="Revolution" height="38" width="38"> </img>  </h1>

[![Generic badge](https://img.shields.io/badge/Feature_Engineering-NOT_AUTO-red.svg)](https://github.com/Muhammad4hmed/Ghalat-Machine-Learning)
[![Generic badge](https://img.shields.io/badge/Machine_Learning-AUTO-<COLOR>.svg)](https://github.com/Muhammad4hmed/Ghalat-Machine-Learning) <br>
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.org/project/GML/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.org/project/GML/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.org/project/GML/)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://GitHub.com/Muhammad4hmed/Ghalat-Machine-Learning/issues/)



<b>Tired of training multiple models and then picking the best among them? No worries now! GML is here for you!</b>
<br>
<br>
GML is an automatic machine learning library in python built on top of Scikit-Learn,Keras,XGBoost,LightGBM and Catboost. with this library, you can train your data on multiple machine learning algorithms and a neural network! not only training but scaling the data for normal distribution and after scaling and training, testing the data on validation data (don't worry you don't need to provide validation data. we will extract it from your data). after testing models on validation data, they will be ranked accordingly and you will see which one performs better than other. the first ranked model will be returned (untrained, so you can train it yourself and check results). You already got some models? no problem! pass them to us to make them compete with our models and let see who wins ;-)<br>
  <br>
In future updates many other things will also be automated like hyper parameter tunning, multiple neural networks, other machine learning algorithms and many more cool things!
<br>
<br>
<h2>Install it: </h2> <br>

```python
pip install GML
```

<br>
<a href = "https://pypi.org/project/GML/">https://pypi.org/project/GML</a> 
<br>
<br>
<h2>See GML in Action!!</h2> <br>

  - For Classification tasks <a href="https://github.com/Muhammad4hmed/Ghalat-Machine-Learning/blob/master/GMLClassifier.ipynb">GML Classifier</a>
  - For Regression tasks <a href="https://github.com/Muhammad4hmed/Ghalat-Machine-Learning/blob/master/GMLRegressor.ipynb">GML Regressor</a> 


<br>
<h2>Function description:</h2><br>
These parameters are common in both GMLRegressor and GMLClassifier<br>

    * X 
      Data column excluding the target column. it can either be a pandas dataframe or a numpy array. but please make sure your data doesn't contains missing data or non-numeric data. (clean it before passing)
    * y 
      The targeted column
  Below parameters are optional.
  
    * metric
      metric on which you want to test your model. by default, it is mean-squared-error for regression and accuracy score for classification
    * test_Size 
      size to split your test data, by default = 0.3 (70% training 30% testing)
    * folds (only in GMLClassifier)
      Data will also be validated using KFolds. pass number of folds. by default folds = 5
    * shuffle
      Shuffle the data when spliting for validation. by default = True
    * scaler
      for Scaler pass:  
        'SS' for StandardScalar
        'MM' for MinMaxScalar
        'log' for Log scalar
         None for not scaling
      by default: StandardScalar
    * models
      You got your own models to make them compete with our models? pass them in a list here. default = None
    * neural_net
      Want to train on Neural Networks? Pass 'Yes', default = 'No'
    * epochs
      for neural networks, by default = 10 
    * verbose
      for neural networks, by default = True
      
  Parameter when creating object of GML <br>
  ```python
  models = Ghalat_Machine_Learning(n_estimators=300)
  ```
  * by default n_estimators are 300, you can change it to whatever you want.


As its first version of GML, feel free to give suggestions,ask questions,report bugs etc in issues portion of this repository!<br>
you can directly contact me at: <font color="blue"> m.ahmed.memonn@gmail.com</font>

I haven't uploaded source code yet on this repo. will upload it later after writing comments.
