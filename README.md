# Stick Out Prediction :triangular_flag_on_post:

 The working principle of this code is mainly based on regression. It makes prediction
on the data that we do not know output by using the output values 5,10 and 15 in the
training data we have. First we import the necessary libraries, then we use the numpy
library to call up the dataset we need to use. After the numpy, the data we have is
available as a string, but we need 
oat type data to do regression, so we convert the
data to 
oat. Then, we determine the input output shape in accordance with our data
set and scale the input outputs with the sckit-learn library. After continuing in this
order to complete Keras steps which are the create model, compile model, fit model,
evaluate, model and run prediction, finally the algorithm do prediction and draw the
graps by using the matplotlib library.

In stick out algorithm and in gas flow algorithm, training and prediction is shown
in the same script.Algorithms firrst opens the training csv file and after training opens
another csv file to do prediction. To seperate all you need to do is to save the training
data, import the same libraries in another script page and move the new input output
part to the next page.

**Csv files have limited information and data because of project privacy. This algorithm has been trained with 1 million line of training data in real project.**

Some samples of training and prediction data can be seen below

**Training**

Timestamp;62.98828;17.90039;5

Timestamp;48.82813;16.78162;10

Timestamp;42.96875;17.12585;15

**Prediction**

2019-11-12T10:22:12.1550851Z;53.22536;16.98242

2019-11-12T10:22:12.1551101Z;53.71424;17.01111

2019-11-12T10:22:12.1551351Z;48.82433;19.78162


