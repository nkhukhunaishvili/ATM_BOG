# ATM_BOG

The motivation for my code is an AWS model described here:
https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

I have tried to understand the model of AWS. They used LSTM for forecasting. The predictors included in the model are: i) lagged valus of times series, ii) derived variables like day of the week, week of the year, etc., plus iii) item id. As I understand, they boil everything together - do not have separate time series for each item. They have a single time series with all the items. The paper behind their model is apploaded in this repository.

I tried to "replicate" their approach. In my model (LSTM) target variable - withdrawal amount y(t) is predicted with lagged valu of withdrawal - y(t-1) and some derived features: day of he week, day of the month and month of the year. These features are one-hot encoded before inclusion. Cashpoint ID is not included in the model. They can be grouped in some categoreis (maybe according to the target variable) and fed into the model as a dummy. The model will also imorprove if more time lags are included - y(t-7), y(of the previous month).

The graphs uploaded show predictions for one of the atms (cashpointid=14) when total withdrawal amount of the previous day is know and when it is not (prediction is made 1 month ahead). Comparison of the two gives a hit that previous valus of the withdrawal is very important. All other variables were available for both prediction scenarios. However, spikes are quite vell captured by the derived featurs. For January 2020 predictions, predictions for different atms vary only initially - due to y(t-1), for the end of the month predicted amount (AMT_SCALED) is the same for every atm.

I have built LSTM for the first time. So the model is very far from perfect. But this approach by AWS is worth invetsigating, I think.
