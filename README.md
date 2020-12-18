# ATM_BOG

The motivation for my code is an AWS model described here:
https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

I have tried to understand the model of AWS. They used LSTM for forecasting and use lagged valus of times series, derived variables like day of the week, week of the year, etc., plus item id as the predictors. As I understand, they boil everything together - do not have separate time series for each item. They have a single time series with all the items.

I tried to replicate their approach. The paper behind their model is apploaded in this repository.
