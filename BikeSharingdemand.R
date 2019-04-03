rm(list=ls())
#Accept training and test data from command line
args = commandArgs(trailingOnly=TRUE)

# test if there are two argument: if not, return an error
if (length(args)< 1) 
{
  stop("train data must be supplied (ex:day.csv)", call.=FALSE)
}else if (length(args) == 1) 
{
  train_file = args[1]
}


load_libraries = c("ggplot2","corrgram","caret","rpart","MASS","sampling","DMwR","randomForest")

check_installed = function(x){
  if(!(x %in% row.names(installed.packages()))){
    install.packages(x)
  }
}
#install packages if not installed
lapply(load_libraries,check_installed)


lapply(load_libraries,require,character.only = TRUE)

rm(load_libraries)
rm(check_installed)

wd = getwd()
if (!is.null(wd)) setwd(wd)

data = read.csv(train_file)

head(data)

#Remove dteday,instant columns
#Because instant and dteday is not carrying any info
data = subset(data,select = -c(dteday,instant))

##Explore the data
str(data)

#Convert categorical data from numeric to factor
columns_factors = c('season','yr','mnth','holiday','weekday','workingday','weathersit')

data[,columns_factors] = lapply(data[,columns_factors],factor)

#############Missing value analysis######
sum(is.na(data))#There are no missing values. So no need to impute

#############Data visualization##########
pl1 = ggplot(data,aes(x=cnt)) 
pl1 + geom_histogram(fill="blue" )

pl2 = ggplot(data,aes(x=season,y=cnt,fill=yr))
pl2 + geom_bar(stat="summary",position = "dodge",fun.y = "mean")

pl3 = ggplot(data,aes(x=mnth,y=cnt,fill=yr))
pl3 + geom_bar(stat="summary",position = "dodge",fun.y = "mean")

pl4 = ggplot(data,aes(x=weathersit,y=cnt,fill=yr))
pl4 + geom_bar(stat="summary",position = "dodge",fun.y = "mean")

pl5 = ggplot(data,aes(x=weekday,y=cnt,fill=yr))
pl5 + geom_bar(stat="summary",position = "dodge",fun.y = "mean")

pl4 = ggplot(data)
pl4+geom_smooth(aes(x=temp,y=cnt),model = lm)
pl4+geom_smooth(aes(x=hum,y=cnt),model = lm)
pl4+geom_smooth(aes(x=windspeed,y=cnt),model = lm)



####Outlier Analysis#####
# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(data,is.numeric) #selecting only numeric

numeric_data = data[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
{
 assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "weekday"), data = subset(data))+ 
          stat_boxplot(geom = "errorbar", width = 0.5) +
          geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                       outlier.size=1, notch=FALSE) +
          theme(legend.position="bottom")+
          labs(y=cnames[i],x="weekday")+
          ggtitle(paste("Box plot of weekday for",cnames[i])))
}

## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,gn7,ncol=4)

#df = data
data.frame()
# #loop to remove from all variables
for(i in cnames){
 #print(i)
 val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
 print(length(val))
 data[,i][data[,i] %in% val] = NA
}

data = knnImputation(data, k = 3)


#As registered and casual data will not be present 
#as input for prediction. These are considered dependant variables
#and removed for model building
data = subset(data,select = -c(registered,casual))

#####Feature Selection########
## Correlation Plot 
corrgram(data, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


## Chi-squared Test of Independence
factor_index = sapply(data,is.factor)
factor_data = data[,factor_index]

chitestres = matrix(nrow = 7,ncol = 7)

for (i in 1:length(factor_data))
{
  for (j in 1:length(factor_data))
  {
    chitest = chisq.test(table(factor_data[,i],factor_data[,j]))
    if(chitest$p.value < 0.05 && i!=j)
    {
      chitestres[i,j] = chitest$p.value
    }
  }
}

chitest = data.frame(chitestres,row.names = colnames(factor_data))
colnames(chitest) = colnames(factor_data)

#Remove columns
data = subset(data,select = -c(atemp,workingday))


####sampling#####
tr.idx = createDataPartition(data$cnt,p=0.8,list=FALSE)
train = data[tr.idx,]
test = data[-tr.idx,]

#MAPE
#calculate MAPE
MAPE = function(y, pred){
  mean(abs((y - pred)/y))
}

#calculate RMSE
RMSE = function(y,pred){
  sqrt(mean((y-pred)^2))
}

#Linear Regression
#check multicollearity
library(usdm)
vif(data[,7:9])

vifcor(data[,7:9], th = 0.9)

#run regression model
lm_model = lm(cnt ~., data = train)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test[,1:9])

qplot(x = test[,10], y = predictions_LR, data = test, color = I("blue"), geom = "point")

#Calculate MAPE
MAPE(test[,10], predictions_LR)

#Calculate RMSE
RMSE(test[,10], predictions_LR)

#Calculate other error metrics
regr.eval(test[,10],predictions_LR,stats = c("mae","rmse","mape","mse"))


################Random Forest###############
RF_model = randomForest(cnt ~ ., train, importance = TRUE, ntree = 500)

summary(RF_model)

#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,-10])

qplot(x = test[,10], y = RF_Predictions, data = test,colour = I("blue"), geom = "point")

#Calculate MAPE
MAPE(test[,10], RF_Predictions)

#Calculate RMSE
RMSE(test[,10],RF_Predictions)

#Calculate other error metrics
regr.eval(test[,10],RF_Predictions,stats = c("mae","rmse","mape","mse"))


#Sample outputs
sample_output = data.frame(test[,10],predictions_LR,RF_Predictions)
colnames(sample_output) = c("true_values","Linear regression pred","Random forest pred")
write.csv(sample_output,"sample_output.csv",row.names = FALSE)
