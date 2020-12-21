# Load necessary packages
library(keras)
library(tensorflow)
library(tidyverse)
library(dplyr)
library(lubridate)
library(ggplot2)
library(ggfortify)
library(tibble)
library(fastDummies)

#Load the data from the working directory
df<-read.csv("train.csv")



#Split OPERATIONDATE into DATE and TIME variables.
#Aggregate the amount for atm and date combination.
#Plus add variables which show day of the week, day of the month and 
#month of the year. 
df<-df %>% separate(OPERATIONDATE, c('DATE','TIME'), sep= " ") %>%
  group_by(CASHPOINTID, DATE) %>% summarise(AMT_SCALED_A=sum(AMT_SCALED)) %>%
  mutate(DATE=ymd(DATE)) %>%
  mutate(WDAY=wday(DATE), MDAY=day(DATE), MYEAR=month(DATE))

#WYEAR=week(DATE), QYEAR=quarter(DATE)

#Add a variable indicating easter holidays in Georgia.
#Other holidays are not included as only easter has 
#moving date. 
easter<-c('2017-04-14','2017-04-15','2017-04-16', '2017-04-17',
          '2018-04-06','2018-04-07','2018-04-08',  '2018-04-09',
          '2019-04-26', '2019-04-27', '2019-04-28', '2019-04-29',
          '2020-04-17','2020-04-18','2020-04-19','2020-04-20')
easter<-ymd(easter)
df$EASTER<-df$DATE %in% easter %>% if_else(1,0)



#EDA
df %>% filter(CASHPOINTID=='14' & year(DATE)=='2019') %>%
  ggplot(aes(WDAY, AMT_SCALED_A)) +
  geom_histogram(stat='identity', binwidth = 1)
#The distribution of overall withdrowals are different for different atms and years.  
#The atms also share some characteristics, such as
#on Monday demand tends to be substantially lower, similar is true for Saturdays.
#In the middle of the week - on Wednesday, demand is tipically comparatively low.
#1 stands for Sunday.

##Knowing if there is any weekly, monthly or annul pattern
#will help in determinig lags to include in the model.
#Looking at monthly series, we see that some shapes of the same length
#(although not very similar) are repeated. - weekly pattern.
a<-df %>% filter(CASHPOINTID==5 & year(DATE)=='2017' & month(DATE)=='1') %>% select(AMT_SCALED_A, DATE)
a %>% ggplot(aes(as.Date(DATE), AMT_SCALED_A))+ 
  geom_line() +
  scale_x_date(date_labels = "%b (%a)")+
  xlab("Date")+
  ylab("Scaled Withdrawal Amount")+
  ggtitle("Weekle Pattern")

#Pattern in months?              
a<-df %>% filter(CASHPOINTID==14 & year(DATE)=='2017') %>% select(AMT_SCALED_A, DATE)
a %>% ggplot(aes(as.Date(DATE), AMT_SCALED_A))+ 
  geom_line() +
  scale_x_date(date_labels = "%Y (%b)")
#In January and first half of February,
#there seems to be lower activity. Which is followed by 
#the most active period of the year - second half of February and March.
#This is a tipical situation for atms but is not true for all of them.
#So there is annual pattern. 



############################################################
#The model will predict y(t) with the help of explanatory 
#variables - y(t-1) and derived features. It would be better
#to include y(t-7), y(of the previous month) as well.

# As I know, stationarity is less of a concern for LSTM,
# So I use data in the original form. 

#Create lagg of target variables as predictor - y(t-1)
df<-df %>% arrange(CASHPOINTID, DATE)
df<-df %>% group_by(CASHPOINTID) %>%
  mutate(LAG1=lag(AMT_SCALED_A, n=1))
#Substitute NA's with 0
df[is.na(df)]<-0


#Create dummy variables from derived features.
df <- dummy_cols(df, select_columns = c('WDAY', 'MDAY','MYEAR'),
                      remove_selected_columns = TRUE)

#split data into train and test sets.
#Use last month from every atm as the test set. Use everything else as the training set.  
test<-df %>% filter (month(DATE)=='12' & year(DATE)=='2019')
train<- df %>% filter(!(month(DATE)=='12' & year(DATE)=='2019'))


#As the data is already scaled, I do not rescale it in any other way.
#One hot encoding is uded for all the categorical data.


y_train = train[, "AMT_SCALED_A"]
x_train = train[, c(-1:-3)]

y_test = test[, "AMT_SCALED_A"]
x_test = test[, c(-1:-3)]

#Training set
x_train<-array(as.matrix(x_train), dim=c(nrow(x_train),1, ncol(x_train)))

y_train <- as.matrix(y_train, nrow = nrow(y_train), ncol = ncol(y_train))

#Validation set
x_val <- array(as.matrix(x_test), 
               dim = c(nrow(x_test), 1, ncol(x_test)))
y_val <- as.matrix(y_test, nrow = nrow(y_test), ncol = ncol(y_test))



#=========================================================================================
####Model training
#Some of the recommended hyperparameters are used here (by AWS).
#"num_cells": "40",+
#"epochs": "20",+
#"learning_rate": "0.001",+
#"num_layers": "3",+
#"dropout_rate": "0.05",+
#"likelihood": "gaussian", - 
#"mini_batch_size": "32",+
#"early_stopping_patience": "10" +

#Number of training sample rows should be multiple of batch size. 
#Number of test set rows should also be multiple of batch size.
#There is a way to get out of this difficulty by saving the model weights
#and assigning them to the model with different batch size. This is done
#later in the code.

batch_size<-126
timesteps<-1
data_dim<-dim(x_train)[3]


model <- keras_model_sequential() 
model %>% layer_lstm(units=40, 
                     batch_input_shape = c(batch_size, timesteps, data_dim),
                     activation='relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 20, activation='relu') %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)
#return_sequences = TRUE, stateful = TRUE
#A stateful recurrent model is one for which the internal states (memories) 
#obtained after processing a batch of samples are reused as initial states for 
#the samples of the next batch. This allows to process longer sequences while 
#keeping computational complexity manageable.


#Accuracy metrics are calculated on training and validation dataset for each epoch.
#The accuracy measures are not used for model training.
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.001, decay = 1e-6 ),  
  metrics = c( 'mae')
)


#Fit the model
#Notice that shuffle=TRUE. Order of the data is not preserved while sampling. 

#Include early stopping criteria.
#https://keras.rstudio.com/reference/callback_early_stopping.html
#https://cran.r-project.org/web/packages/keras/vignettes/training_callbacks.html

Epochs = 20
start_time <- Sys.time()

#I have set patience=0 to speed up the training. 
for(i in 1:Epochs ){
  model %>% fit(x=x_train, y=y_train, epochs=20, batch_size=126,  verbose=1, shuffle=TRUE,
                callback_early_stopping(
                  monitor = "loss",
                  patience = 0,
                  mode = c("min"),
                  restore_best_weights = TRUE
                ))
  model %>% reset_states()
}

end_time <- Sys.time()
end_time - start_time
#Took 18 minutes

#Save the model
model %>% save_model_hdf5("model_2.h5")


#Change batch_size in the model
batch_size<-27893
timesteps<-1
data_dim<-dim(x_train)[3]


model_2 <- keras_model_sequential() 
model_2 %>% layer_lstm(units=40, 
                     batch_input_shape = c(batch_size, timesteps, data_dim),
                     activation='relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 20, activation='relu') %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

model_2 <- load_model_hdf5("model_2.h5")
summary(model_2)


#Use the weights of the saved model.
#https://keras.rstudio.com/articles/saving_serializing.html
#weights<-get_weights(model_2)
#set_weights(model_new, weights)


#Make the prediction for the test set using the whole test set
predictions<-model_2 %>% predict(x_val, batch_size=27893)


#Save the predictions
b<-cbind(test, pred=as.vector(predictions))
write.csv(b,"predictions_2.csv")


#Compute root mean squared error on test set.
#b<-read.csv("predictions_2.csv")
sqrt(sum((b$pred-test$AMT_SCALED_A)^2))
#RMSE=19.05


#Make a plot to compare predicted vs actual withdrawals by atm.
b<-cbind(test, pred=as.vector(predictions))
 b %>% head
b %>% filter(CASHPOINTID=='1301' & year(DATE)=='2019') %>%
  ggplot()+
  geom_line(aes(x=as.Date(DATE), y=pred, color='Predicted'))+
  geom_line(aes(x=as.Date(DATE), y=AMT_SCALED_A, color='Actual'))+
  xlab("Date")+
  ylab("Scaled Withdrawal Amount")+
  ggtitle("Prediction for December-2019")+
  scale_color_manual(name = "Comparison", values = c("Predicted"="#F25022", "Actual"="#007CBA"))
  
#Predictions for some of the atms are pretty good.
#Some are very bad. Plus
#these predictions are computed as though I know 
#total value withdrawn from the atm the previous day.


#To make test set similar to the problem, will leave only November 30
#values of atm withdrawal and delete everything else.

#Make predictions for 1 month ahead.
#Make 1 step prediction and use it for the next prediction.

#Update the batch_size again. Set it to 1.
batch_size<-1
timesteps<-1
data_dim<-dim(x_train)[3]


model_2 <- keras_model_sequential() 
model_2 %>% layer_lstm(units=40, 
                     batch_input_shape = c(batch_size, timesteps, data_dim),
                     activation='relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 20, activation='relu') %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)


#Predict 1 step at a time
#In the test set leave only values of lagg for December 1 - this is
#all I know when making prediction on November 30.
#test<-df %>% filter (month(DATE)=='12' & year(DATE)=='2019')
test_mod=test
#The data frame is already ordered by cashpoint id and date
test_mod$LAG1[!(day(test$DATE)=='1')]=NaN
#If any of the lag value on December 1 is NaN, set it to 0.
#Currently there is no such entry.
test_mod %>% filter(day(DATE)=='1' & is.na(LAG1))



y_test = test_mod[, "AMT_SCALED_A"]
x_test = test_mod[, c(-1:-3)]

L = nrow(x_test)
predictions_1step = numeric(L)

start_time <- Sys.time()

for(i in 1:L){
  X=as.matrix(x_test[i,])
  X <- array(X, dim = c(1, 1, ncol(x_test)))
  yhat = model_2 %>% predict(X, batch_size=1)
  # store
  predictions_1step[i] <- yhat
  if (is.na(x_test[i+1,"LAG1"])) {x_test[i+1,"LAG1"]<-yhat}
}

end_time <- Sys.time()
end_time - start_time
#Took 17 minutes

#Make a plot to compare predicted vs actual withdrawals by atm.
c<-cbind(test, pred=as.vector(predictions_1step))
c %>% head
c %>% filter(CASHPOINTID=='14' & year(DATE)=='2019') %>%
  ggplot()+
  geom_line(aes(x=as.Date(DATE), y=pred, color='Predicted'))+
  geom_line(aes(x=as.Date(DATE), y=AMT_SCALED_A, color='Actual'))+
  xlab("Date")+
  ylab("Scaled Withdrawal Amount")+
  ggtitle("Prediction for December-2019 - No History")+
  scale_color_manual(name = "Comparison", values = c("Predicted"="#F25022", "Actual"="#007CBA"))
#Save the predictions
#write.csv(c,"predictions_1step.csv")




#=======================================================================================
#Make out of sample predictions
#Create dataset with CASHPOINTID and DATE
ATM<-0:1342
D<-seq(as.Date("2020-01-01"), as.Date("2020-01-31"), by="days")
x_out<-expand.grid(ATM,D)
names(x_out)<-c("CASHPOINTID","DATE")


#Creat derived features
x_out<-x_out %>%
  mutate(DATE=ymd(DATE)) %>%
  mutate(WDAY=wday(DATE), MDAY=day(DATE), MYEAR=month(DATE))
x_out$EASTER<-x_out$DATE %in% easter %>% if_else(1,0)

#Transform them to dummy variables
x_out <- dummy_cols(x_out, select_columns = c('WDAY', 'MDAY','MYEAR'),
                 remove_selected_columns = TRUE)

#From the initial dataframe take the last observations to serve as predictiors
filtered<-df %>% filter(DATE=='2019-12-31') %>% select(CASHPOINTID, DATE,AMT_SCALED_A)

#Join the filtered table with x_out table to
#get values for the 31st of December 2019.
x_out$DATE=as.Date(x_out$DATE)-1
x_out=left_join(x_out, filtered, by=c("CASHPOINTID", "DATE"))
colnames(x_out)[43] <- "LAG1"
colnames(x_out)[42] <- "MYEAR_12"
x_out$DATE=as.Date(x_out$DATE)+1
#Wherever prevous day value is na, set it to 0.
x_out$LAG1[x_out$DATE=='2020-01-01' & is.na(x_out$LAG1)]=0

#Dummy variable should be same as in the training set
x_out$MYEAR_11<-rep(0, length=nrow(x_out))

#For correct prediction data should be ordered by time
#for every atm.
x_out<-x_out %>% arrange(CASHPOINTID, DATE)
x_out_a<-x_out

#Delete unnecessary columns
x_out=x_out[,-1:-2]

#Reorder features
col_order<-c(names(x_test))
x_out<-x_out[,col_order]

#Save x_out
#write.csv(x_out, "x_out.csv")

#Predict
L = nrow(x_out)
predictions_out = numeric(L)

start_time <- Sys.time()

for(i in 1:L){
  X=as.matrix(x_out[i,])
  X <- array(X, dim = c(1, 1, ncol(x_out)))
  yhat = model_1 %>% predict(X, batch_size=1)
  # store
  predictions_out[i] <- yhat
  if (is.na(x_out[i+1,"LAG1"])) {x_out[i+1,"LAG1"]<-yhat}
}

end_time <- Sys.time()
end_time - start_time
#25 minutes

x_out<-x_out[-41634,] 


#Save the predictions
#write.csv(predictions_out,"predictions_out.csv")
o<-cbind(x_out_a, pred=as.vector(predictions_out))
write.csv(o,'predictions_out.csv')

#Plot the predictions
o %>% filter(CASHPOINTID=='0') %>%
  ggplot()+
  geom_line(aes(x=as.Date(DATE), y=pred, color='Predicted'))+
  xlab("Date")+
  ylab("Scaled Withdrawal Amount")+
  ggtitle("Prediction for December-2019")+
  scale_color_manual(name = "Comparison", values = c("Predicted"="#F25022", "Actual"="#007CBA"))

final_output<-o[,c("CASHPOINTID", "DATE","pred")]
names(final_output)<-c("CASHPOINTID", "OPERATIONDATE","AMT_SCALED")
write.csv(final_output, "output.csv")



