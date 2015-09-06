A learner to forecast 5 day future returns of stock prices.
The project creates a learning system that reads many files of historical time series data, and then builds a model from that data that would predict 5 day returns. 
I used Random Forest Learner from sklearn.ensemble, and use amplitude, phase, standard deviation and the first derivative as my features.