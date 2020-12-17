#Motivation for this code:
#https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html
# Load the necessary packages
library(keras)
library(tensorflow)
library(tidyverse)
library(dplyr)
library(lubridate)
library(ggplot2)
library(ggfortify)
library(tibble)

#Load the data from the working directory
df<-read.csv("train.csv")
df %>% head()


#Split OPERATIONDATE into DATE and HOUR variables
#Aggregate the amount for atm and date combination
#Plus add variables which show day of the week, day of the month,
#day of the year, month of the year, week of the year and
df<-df %>% separate(OPERATIONDATE, c('DATE','HOUR'), sep= " ") %>%
  group_by(CASHPOINTID, DATE) %>% summarise(AMT_SCALED_A=sum(AMT_SCALED)) %>%
  mutate(DATE=ymd(DATE)) %>%
  mutate(WDAY=wday(DATE), MDAY=day(DATE), YDAY=yday(DATE), MONTH=month(DATE),
         YWEEK=week(DATE))

#Add variable which indicates whether it is holiday in Georgia or not
#Source: https://www.nplg.gov.ge/?m=205
#holiday_GE<-function(x_date) {
#  x_date<-ymd(x_date)
  #The function returns True if the date in the argument is
  #a special holiday (like easter) in Georgia.
  #csv file named easter needs to be preloaded.
  #a<-0
#  if (
    #January holidays
#    (month(x_date)==1 & 
#     (day(x_date)==1 | day(x_date)==2 
#      | day(x_date)==7 | day(x_date)==19)) |
    #March holidays
#    (month(x_date)==3 & 
#     (day(x_date)==3 | day(x_date)==8)) |
#    #April holidays
#    (month(x_date)==4 & day(x_date)==9) |
    #May holidays
#    (month(x_date)==5 & 
#     ((day(x_date)==9) | day(x_date)==12 | 
#      day(x_date)==26)) |
#    #August holidays
#    (month(x_date)==8 & day(x_date)==28) |
#    #October holidays
#    (month(x_date)==10 & day(x_date)==14) |
#    #November holidays
#   (month(x_date)==11 & day(x_date)==23) |
#    #Easter holidays
#    x_date %in% (dmy(easter$orthodox)-2)  |
#    x_date %in% (dmy(easter$orthodox)-1) |
#    x_date %in% dmy(easter$orthodox) |
#    x_date %in% (dmy(easter$orthodox)+1)
#  ) {a<-1}
#  return(a) 
#}

#df$HOLIDAY<-sapply(df$DATE, FUN = holiday_GE)

#Save the dataset
#write.csv(df, "train_a.csv")

############################################################
#The model I build, will have predict y(t) with the help of explanatory 
#variables - y(t-1), day of the week...week of the year, plus
#CASHPOINTID will be one of them.

#Changing the shape of the dataset to have amt for every ATM in
#separate columns reveals Some of them seems to be new as
#older data is missing from them.

#Although in according to AWS approach missing values would be replaced with 0,
#I do nothing with 'missjng values'. 
#df[is.na(df)]<-0

# transform data to become stationary by first differencing
df<-df %>% group_by(CASHPOINTID) %>%
  mutate(DIFFED=AMT_SCALED_A-lag(AMT_SCALED_A))
#Delete the rows with NA
df<-df[-is.na(df),]

#Create another variable - y(t-1)
df<-df %>% group_by(CASHPOINTID) %>%
  mutate(LAGGED=lag(DIFFED))
#Substitute NA's with 0
df[is.na(df)]<-0



## split data into train and test sets in an unusual way for time series
#I realized this is not the best way of splitting.
#Reversing the first differencing becomes much more complicated.
df$id <- 1:nrow(df)
train <- df %>% dplyr::sample_frac(.8)
test  <- dplyr::anti_join(df, train, by = 'id')


#As the data is already scaled, I do not rescale it in any other way.
y_train = train[, c("DIFFED")]
x_train = train[, c("LAGGED",'CASHPOINTID', "WDAY", "MDAY", "YDAY", "MONTH", "YWEEK")]

y_test = test[, c("DIFFED")]
x_test = test[, c("LAGGED",'CASHPOINTID', "WDAY", "MDAY", "YDAY", "MONTH", "YWEEK")]

x_train<-x_train %>% as.matrix()
y_train<-y_train %>% as.matrix()

batch_size<-10


dim(x_train)<-c(nrow(x_train), 1, ncol(x_train))
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]


####Modeling
#=========================================================================================

model <- keras_model_sequential() 
model%>%
  layer_lstm(units=1, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE) %>%
  layer_dense(units = 1)



model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)


#Fit the model
Epochs = 10  
for(i in 1:Epochs ){
  model %>% fit(x=x_train, y=y_train, epochs=1, batch_size=10, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


#Save the model
model %>% save_model_hdf5("my_model.h5")
my_model <- load_model_hdf5("my_model.h5")


#Do not have time to test on test set


#Make out of sample prediction
ATM<-0:1342
D<-seq(as.Date("2020-01-01"), as.Date("2020-01-31"), by="days")
pred<-expand.grid(ATM,D)
names(pred)<-c("CASHPOINTID","DATE")

#Creat additional variables
pred<-pred %>% 
  mutate(DATE=ymd(DATE)) %>%
  mutate(WDAY=wday(DATE), MDAY=day(DATE), YDAY=yday(DATE), MONTH=month(DATE),
         YWEEK=week(DATE))

#From the initial dataframe take the last observations to serve as predictiors
last<-data.frame(CASHPOINTID=0:1342)
filtered<-df %>% filter(DATE=='2019-12-31')
last<-left_join(last,filtered, by="CASHPOINTID" )
last$DATE=ymd("2020-01-01")


last<-last[,c(1,2,3,9)]

pred<-left_join(pred, last, by=c("CASHPOINTID", "DATE"))
names(pred)<-c("CASHPOINTID", "DATE", "WDAY", "MDAY", "YDAY", "MONTH",      
               "YWEEK", "AMT_SCALED_A", "LAGGED" )


#Predict

#I am not sure that group_by() produces correct predictions in this case
#Prediction of current day needs to be predictor of the next day
#for each ATM
#Pred is sorted by DATE

pred %>% group_by(CASHPOINTID)

x_pred = pred[, c("LAGGED",'CASHPOINTID', "WDAY", "MDAY", "YDAY", "MONTH", "YWEEK")]


L = nrow(x_pred)
predictions = numeric(L)

for(i in 1:L){
  X = as.matrix(x_pred[i,])
  dim(X) = c(1,1,7)
  yhat = my_model %>% predict(X, batch_size=1)
  # store
  predictions[i] <- yhat
}


# invert differencing

