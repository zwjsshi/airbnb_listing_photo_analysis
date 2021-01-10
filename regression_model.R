ny_bed_final = read.csv('C:\\Users\\zewen\\Desktop\\Computer Vision Marketing\\Project\\final\\ny_bed_final.csv')
names(ny_bed_final)
attach(ny_bed_final)
###correlation Matrix
library(dplyr)
numerical_vars_with_target = dplyr::select(ny_bed_final, 6:48)
library(ggplot2)
library(ggfortify)
library(GGally)
ggcorr(numerical_vars_with_target)+scale_fill_gradient2(low="light green",high="dark blue",mid="gray91")

###PCA for numerical features 
#check constant 
which(apply(numerical_vars_with_target, 2, var)==0)
numerical_vars_with_target = numerical_vars_with_target[ , apply(numerical_vars_with_target, 2, var) != 0]
pca1 = prcomp(numerical_vars_with_target, scale=TRUE)
autoplot(pca1,data=numerical_vars_with_target,loadings=TRUE,col='grey',loadings.label=TRUE)
pca1

library(randomForest)
###random forest importance for non-numerical features
myforest=randomForest(Price~Neighbourhood.Cleansed+Review.Scores.Rating+Bedrooms+Beds+Number.of.Reviews
                      +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
                      +crib+red+green+blue+brightness+corners+edges+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
                      +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6+dim7
                      +dim8+dim9+dim10+dim11+dim12+dim13,ntree=10000, data=ny_bed_final,importance=TRUE)
importance(myforest)
varImpPlot(myforest)


###linear regression model built
#remove bedrooms
lreg1 = lm(Review.Scores.Rating~as.factor(Neighbourhood.Cleansed)+Price+Beds+Number.of.Reviews
          +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
          +crib+red+green+blue+brightness+corners+edges+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
          +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6+dim7
          +dim8+dim9+dim10+dim11+dim12+dim13)
#analyze the regression1
plot(lreg1,col=terrain.colors(4))
summary(lreg1)

#remove non-related features (NA): edges, dim7
lreg2 = lm(Review.Scores.Rating~as.factor(Neighbourhood.Cleansed)+Price+Beds+Number.of.Reviews
           +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
           +crib+red+green+blue+brightness+corners+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
           +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6
           +dim8+dim9+dim10+dim11+dim12+dim13)
#analyze the regression2
plot(lreg2,col=terrain.colors(10))
summary(lreg2)

###non-linearity test
library(car)
residualPlots(lreg2, col = terrain.colors(10))
plot(predict(lreg2), residuals(lreg2), col=terrain.colors(10))
abline(0,0, lty=2)
#non-linear features: beds, dim8, dim13 
#Tukey test: 0.051 (p-value) => almost linear model 

###outlier test
#p-value < 0.05 are reported as outliers
outlierTest(lreg2)

###new dataset without the outliers
ny_bed_final_no_out = ny_bed_final[-c(1594,1343,1354,426,743,926,654,1537,1153,721),]#
detach(ny_bed_final)
attach(ny_bed_final_no_out)
#remove outliers
lreg3 = lm(Review.Scores.Rating~as.factor(Neighbourhood.Cleansed)+Price+Beds+Number.of.Reviews
           +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
           +crib+red+green+blue+brightness+corners+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
           +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6
           +dim8+dim9+dim10+dim11+dim12+dim13)
#analyze the regression3
plot(lreg3,col=terrain.colors(4))
summary(lreg3)

###Recheck nonlinearity
residualPlots(lreg3, col = terrain.colors(10))
plot(predict(lreg3), residuals(lreg3), col=terrain.colors(10))
abline(0,0, lty=2)
#non-linear: beds, dim8, fourposter
#Tukey test: 0.70
#for now, i didn't try polynomial regression, so we use lreg3 

###prediction 
#prediction_test = read.csv("C:\\...\\Test.csv")
prediction_test = dplyr::sample_n(ny_bed_final_no_out, 100, replace=TRUE)
#get the imageID for those random rows
test_result = prediction_test$Review.Scores.Rating
detach(ny_bed_final_no_out)
attach(prediction_test)
#predict
predicted_rating = predict(lreg3,newdata=prediction_test,type='response')
predicted_rating
#MSE
MSE = sum((test_result - predicted_rating)^2)/nrow(prediction_test)
MSE

#write.csv(predicted_imdbrating, "C:\\...\\Prediction.csv")



detach(ny_bed_final_no_out)
attach(ny_bed_final)
###linear regression model built
#remove price, bedrooms
lreg4 = lm(Price~as.factor(Neighbourhood.Cleansed)+Review.Scores.Rating+Beds+Number.of.Reviews
           +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
           +crib+red+green+blue+brightness+corners+edges+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
           +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6+dim7
           +dim8+dim9+dim10+dim11+dim12+dim13)
#analyze the regression1
plot(lreg1,col=terrain.colors(4))
summary(lreg4)

#remove non-related features (NA): edges, dim7
lreg5 = lm(Price~as.factor(Neighbourhood.Cleansed)+Review.Scores.Rating+Beds+Number.of.Reviews
           +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
           +crib+red+green+blue+brightness+corners+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
           +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6
           +dim8+dim9+dim10+dim11+dim12+dim13)
#analyze the regression2
plot(lreg2,col=terrain.colors(10))
summary(lreg5)

###non-linearity test
library(car)
residualPlots(lreg5, col = terrain.colors(10))
plot(predict(lreg5), residuals(lreg5), col=terrain.colors(10))
abline(0,0, lty=2)
#non-linear features: beds, corners
#Tukey test: < 2e-16 *** (p-value) => almost non-linear model 

###outlier test
#p-value < 0.05 are reported as outliers
outlierTest(lreg5)

###new dataset without the outliers
ny_bed_final_no_out2 = ny_bed_final[-c(735,17,1538,645,8,326,1361,1232,557,595),]#
detach(ny_bed_final)
attach(ny_bed_final_no_out2)
#remove outliers & corners
lreg6 = lm(Price~as.factor(Neighbourhood.Cleansed)+Review.Scores.Rating+Beds+Number.of.Reviews
           +fourposter+studiocouch+quilt+mosquitonet+wardrobe+slidingdoor+windowshade+shoji+hometheater
           +crib+red+green+blue+brightness+lines+hog_0.20+hog_20.40+hog_40.60+hog_60.80
           +hog_80.100+hog_100.120+hog_120.140+hog_140.160+dim1+dim2+dim3+dim4+dim5+dim6
           +dim8+dim9+dim10+dim11+dim12+dim13)
#analyze the regression3
plot(lreg3,col=terrain.colors(4))
summary(lreg6)

###Recheck nonlinearity
residualPlots(lreg6, col = terrain.colors(10))
plot(predict(lreg6), residuals(lreg6), col=terrain.colors(10))
abline(0,0, lty=2)
#non-linear: beds, dim6
#Tukey test: 6.217e-15 ***
#for now, i didn't try polynomial regression, so we use lreg3 

###prediction 
#prediction_test = read.csv("C:\\...\\Test.csv")
prediction_test2 = dplyr::sample_n(ny_bed_final_no_out2, 100, replace=TRUE)
#get the imageID for those random rows
test_result2 = prediction_test2$Price
detach(ny_bed_final_no_out2)
attach(prediction_test2)
#predict
predicted_pricing = predict(lreg6,newdata=prediction_test2,type='response')
predicted_pricing
#MSE
MSE2 = sum((test_result2 - predicted_pricing)^2)/nrow(prediction_test2)
MSE2








