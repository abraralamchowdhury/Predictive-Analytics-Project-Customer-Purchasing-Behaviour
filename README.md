# Predictive-Analytics-Project-Customer-Purchasing-Behaviour
Customer Purchasing Behaviour
MDA611 Assignment 2 — Predictive Analytics
02 September 2025
1 Cover Page
2 Task 0 — Setup
3 Task 1 — Pre-processing (Cleaning, Ethics, Feature Engineering)
4 Task 2 — Exploratory Data Analysis (EDA)
5 Task 3 — EDA findings and model justification
6 Task 4 — Modelling
6.1 Model 1 — Random Forest
6.2 Model 2 — XGBoost
7 Task 5 — Evaluation
8 Task 6 — Model improvement
9 Task 7 — Comparison and conclusion
10 Session Info
1 Cover Page
Unit: MDA611 Predictive Analytics
Assessment: Assignment 2 (Group)
Dataset: shopDataAssignment2.csv (≈3900 rows × 18 columns)
Target: Frequency.of.Purchases (multi-valued classification)

2 Task 0 — Setup
set.seed(42)

install_if_missing <- function(pkgs) {
  to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)
}

install_if_missing(c(
  "tidyverse","data.table","janitor","lubridate","skimr",
  "caret","ranger","xgboost","e1071","forcats","ggplot2",
  "yardstick","rpart","rpart.plot","vip","ggrepel","stringr","GGally"
))

library(tidyverse)
library(data.table)
library(janitor)
library(lubridate)
library(skimr)
library(caret)
library(ranger)
library(xgboost)
library(e1071)
library(forcats)
library(ggplot2)
library(vip)
library(ggrepel)
library(stringr)
library(GGally)

options(width = 120)
theme_set(theme_minimal())
DATA_CSV <- "shopDataAssignment2.csv"
stopifnot(file.exists(DATA_CSV))
3 Task 1 — Pre-processing (Cleaning, Ethics, Feature Engineering)
dt_raw <- fread(DATA_CSV)
df <- as.data.frame(dt_raw)
df <- clean_names(df)

glimpse(df)
## Rows: 3,902
## Columns: 21
## $ unnamed_0                <dbl> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24…
## $ age                      <dbl> 55, 19, 50, 21, 45, 46, 63, 27, 26, 57, 53, 30, 61, 65, 64, 64, 25, 53, 52, 66, 21, 3…
## $ gender                   <chr> "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male…
## $ item_purchased           <chr> "Blouse", "Sweater", "Jeans", "Sandals", "Blouse", "Sneakers", "Shirt", "Shorts", "Co…
## $ category                 <chr> "Clothing", "Clothing", "Clothing", "Footwear", "Clothing", "Footwear", "Clothing", "…
## $ purchase_amount_usd      <dbl> 53, 64, 73, 90, 49, 20, 85, 34, 97, 31, 34, 68, 72, 51, 53, 81, 36, 38, 48, 90, 51, 6…
## $ size                     <chr> "L", "L", "S", "M", "M", "M", "M", "L", "L", "M", "L", "S", "M", "M", "L", "M", "S", …
## $ color                    <chr> "Gray", "Maroon", "Maroon", "Maroon", "Turquoise", "White", "Gray", "Charcoal", "Silv…
## $ season                   <chr> "Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Fall", "Winter", "Summer…
## $ review_rating            <dbl> 3.1, 3.1, 3.1, 3.5, 2.7, 2.9, 3.2, 3.2, 2.6, 4.8, 4.1, 4.9, 4.5, 4.7, 4.7, 2.8, 4.1, …
## $ subscription_status      <chr> "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "…
## $ payment_method           <chr> "Credit Card", "Bank Transfer", "Cash", "PayPal", "Cash", "Venmo", "Debit Card", "Deb…
## $ shipping_type            <chr> "Express", "Express", "Free Shipping", "Next Day Air", "Free Shipping", "Standard", "…
## $ discount_applied         <chr> "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "…
## $ promo_code_used          <chr> "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "…
## $ previous_purchases       <dbl> 14, 2, 23, 49, 31, 14, 49, 19, 8, 4, 26, 10, 37, 31, 34, 8, 44, 36, 17, 46, 50, 22, 3…
## $ preferred_payment_method <chr> "Venmo", "Cash", "Credit Card", "PayPal", "PayPal", "Venmo", "Cash", "Credit Card", "…
## $ frequency_of_purchases   <chr> "Fortnightly", "Fortnightly", "Weekly", "Weekly", "Annually", "Weekly", "Quarterly", …
## $ purchase_date            <chr> "19/06/2025", "7/11/2024", "19/03/2025", "18/08/2024", "23/10/2024", "15/11/2024", "2…
## $ region                   <chr> "South", "Northeast", "Northeast", "Northeast", "West", "West", "West", "South", "Sou…
## $ blank_feature            <lgl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, N…
skimr::skim(df)
Data summary
Name	df
Number of rows	3902
Number of columns	21
_______________________	
Column type frequency:	
character	15
logical	1
numeric	5
________________________	
Group variables	None
Variable type: character

skim_variable	n_missing	complete_rate	min	max	empty	n_unique	whitespace
gender	0	1	4	6	0	2	0
item_purchased	0	1	3	10	0	25	0
category	0	1	8	11	0	4	0
size	0	1	1	2	0	4	0
color	0	1	3	9	0	25	0
season	0	1	4	6	0	4	0
subscription_status	0	1	2	3	0	2	0
payment_method	0	1	4	13	0	6	0
shipping_type	0	1	7	14	0	6	0
discount_applied	0	1	2	3	0	2	0
promo_code_used	0	1	2	3	0	2	0
preferred_payment_method	0	1	4	13	0	6	0
frequency_of_purchases	0	1	6	14	0	7	0
purchase_date	0	1	9	10	0	365	0
region	0	1	4	9	0	4	0
Variable type: logical

skim_variable	n_missing	complete_rate	mean	count
blank_feature	3902	0	NaN	:
Variable type: numeric

skim_variable	n_missing	complete_rate	mean	sd	p0	p25	p50	p75	p100	hist
unnamed_0	0	1	1944.64	1132.52	-781.00	966.25	1943.5	2922.75	5961.84	▃▇▇▃▁
age	0	1	43.92	15.51	-33.00	31.00	44.0	57.00	117.52	▁▂▇▃▁
purchase_amount_usd	0	1	59.50	24.36	-112.73	38.00	60.0	81.00	195.80	▁▁▇▅▁
review_rating	0	1	3.74	0.75	-4.40	3.10	3.7	4.40	7.88	▁▁▂▇▁
previous_purchases	0	1	25.33	14.57	-11.00	13.00	25.0	38.00	98.05	▃▇▆▁▁
df$frequency_of_purchases <- as.factor(df$frequency_of_purchases)
levels(df$frequency_of_purchases) <- make.names(levels(df$frequency_of_purchases))

id_like <- names(df)[str_detect(names(df), regex("id$|^id|customer|user", ignore_case = TRUE))]
id_like <- unique(intersect(id_like, names(df)))

cat_cols <- names(df)[sapply(df, is.character) | sapply(df, is.factor)]
num_cols <- names(df)[sapply(df, is.numeric)]

for (c in setdiff(cat_cols, "frequency_of_purchases")) {
  df[[c]] <- as.factor(df[[c]])
}

collapse_high_card <- function(x, max_levels = 6) {
  if (!is.factor(x)) return(x)
  if (nlevels(x) <= max_levels) return(x)
  fct_lump(x, n = max_levels - 1, other_level = "Other")
}
df <- df %>% mutate(across(.cols = where(is.factor) & !matches("^frequency_of_purchases$"), .fns = ~ collapse_high_card(.x, 6)))

impute_mode <- function(v) {
  v2 <- as.character(v)
  tab <- table(v2, useNA = "no")
  if (length(tab) == 0) return(factor(v))
  m <- names(sort(tab, decreasing = TRUE))[1]
  v2[is.na(v2) | v2 == ""] <- m
  factor(v2)
}
df <- df %>% mutate(across(where(is.numeric), ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))
df <- df %>% mutate(across(where(is.factor) & !matches("^frequency_of_purchases$"), ~ impute_mode(.x)))

top_class <- df %>% count(frequency_of_purchases, sort = TRUE) %>% arrange(desc(n)) %>% slice_head(n = 1) %>% pull(frequency_of_purchases)
df$freq_top_vs_rest <- factor(ifelse(df$frequency_of_purchases == top_class, "Top", "Rest"))

glimpse(df)
## Rows: 3,902
## Columns: 22
## $ unnamed_0                <dbl> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24…
## $ age                      <dbl> 55, 19, 50, 21, 45, 46, 63, 27, 26, 57, 53, 30, 61, 65, 64, 64, 25, 53, 52, 66, 21, 3…
## $ gender                   <fct> Male, Male, Male, Male, Male, Male, Male, Male, Male, Male, Male, Male, Male, Male, M…
## $ item_purchased           <fct> Blouse, Other, Other, Other, Blouse, Other, Shirt, Other, Other, Other, Other, Other,…
## $ category                 <fct> Clothing, Clothing, Clothing, Footwear, Clothing, Footwear, Clothing, Clothing, Outer…
## $ purchase_amount_usd      <dbl> 53, 64, 73, 90, 49, 20, 85, 34, 97, 31, 34, 68, 72, 51, 53, 81, 36, 38, 48, 90, 51, 6…
## $ size                     <fct> L, L, S, M, M, M, M, L, L, M, L, S, M, M, L, M, S, XL, S, M, M, M, M, XL, M, M, M, L,…
## $ color                    <fct> Other, Other, Other, Other, Other, Other, Other, Other, Silver, Other, Other, Olive, …
## $ season                   <fct> Winter, Winter, Spring, Spring, Spring, Summer, Fall, Winter, Summer, Spring, Fall, W…
## $ review_rating            <dbl> 3.1, 3.1, 3.1, 3.5, 2.7, 2.9, 3.2, 3.2, 2.6, 4.8, 4.1, 4.9, 4.5, 4.7, 4.7, 2.8, 4.1, …
## $ subscription_status      <fct> Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, …
## $ payment_method           <fct> Credit Card, Bank Transfer, Cash, PayPal, Cash, Venmo, Debit Card, Debit Card, Venmo,…
## $ shipping_type            <fct> Express, Express, Free Shipping, Next Day Air, Free Shipping, Standard, Free Shipping…
## $ discount_applied         <fct> Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, …
## $ promo_code_used          <fct> Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, Yes, …
## $ previous_purchases       <dbl> 14, 2, 23, 49, 31, 14, 49, 19, 8, 4, 26, 10, 37, 31, 34, 8, 44, 36, 17, 46, 50, 22, 3…
## $ preferred_payment_method <fct> Venmo, Cash, Credit Card, PayPal, PayPal, Venmo, Cash, Credit Card, Venmo, Cash, Bank…
## $ frequency_of_purchases   <fct> Fortnightly, Fortnightly, Weekly, Weekly, Annually, Weekly, Quarterly, Weekly, Annual…
## $ purchase_date            <fct> Other, Other, Other, Other, Other, Other, Other, Other, Other, Other, Other, Other, O…
## $ region                   <fct> South, Northeast, Northeast, Northeast, West, West, West, South, South, Midwest, Sout…
## $ blank_feature            <lgl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, N…
## $ freq_top_vs_rest         <fct> Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, Rest, R…
4 Task 2 — Exploratory Data Analysis (EDA)
# Target distribution
df %>% count(frequency_of_purchases) %>% mutate(prop = n/sum(n)) -> class_dist
class_dist
##   frequency_of_purchases   n      prop
## 1               Annually 572 0.1465915
## 2              Bi.Weekly 547 0.1401845
## 3         Every.3.Months 585 0.1499231
## 4            Fortnightly 543 0.1391594
## 5                Monthly 553 0.1417222
## 6              Quarterly 563 0.1442850
## 7                 Weekly 539 0.1381343
ggplot(class_dist, aes(x = frequency_of_purchases, y = n)) +
  geom_col(fill = "steelblue") + geom_text(aes(label = n), vjust = -0.3) +
  labs(title = "Frequency.of.Purchases distribution", x = "Class", y = "Count")


# Numeric distributions
df %>% select(all_of(num_cols)) %>%
  gather(var, value) %>%
  ggplot(aes(x = value, fill = var)) +
  geom_histogram(bins = 30, alpha = 0.6) +
  facet_wrap(~ var, scales = "free") +
  labs(title = "Numeric variable distributions")


# Correlation heatmap
ggcorr(df[, num_cols], label = TRUE, hjust = 1, size = 3) +
  labs(title = "Correlation heatmap of numeric variables")


# Categorical vs Target (example with first two categorical predictors)
for (col in head(setdiff(cat_cols, "frequency_of_purchases"), 2)) {
  print(
    ggplot(df, aes_string(x = col, fill = "frequency_of_purchases")) +
      geom_bar(position = "fill") +
      labs(title = paste("Distribution of", col, "by Purchase Frequency"), y = "Proportion")
  )
}
## Warning: `aes_string()` was deprecated in ggplot2 3.0.0.
## ℹ Please use tidy evaluation idioms with `aes()`.
## ℹ See also `vignette("ggplot2-in-packages")` for more information.
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated.


5 Task 3 — EDA findings and model justification
Random Forest is being chosen for handling mixed data types and interpretability.
XGBoost is being chosen for performance with boosted trees.
6 Task 4 — Modelling
preds_all <- setdiff(names(df), c("frequency_of_purchases", "freq_top_vs_rest", id_like))

set.seed(42)
train_idx <- createDataPartition(df$frequency_of_purchases, p = 0.8, list = FALSE)
train_df <- df[train_idx, , drop = FALSE]
test_df  <- df[-train_idx, , drop = FALSE]

for (v in names(train_df)) {
  if (is.factor(train_df[[v]])) {
    test_df[[v]] <- factor(test_df[[v]], levels = levels(train_df[[v]]))
  }
}

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 2, classProbs = TRUE, savePredictions = "final")


# removing near-zero variance predictors to avoid caret xgbTree errors
nzv <- nearZeroVar(train_df[, preds_all, drop = FALSE])
if (length(nzv) > 0) {
  preds_all <- preds_all[-nzv]
}
6.1 Model 1 — Random Forest
rf_grid <- expand.grid(mtry = max(1, floor(sqrt(length(preds_all)))), splitrule = "gini", min.node.size = 5)

set.seed(42)
fit_rf <- train(
  x = train_df[, preds_all, drop = FALSE],
  y = train_df$frequency_of_purchases,
  method = "ranger",
  trControl = ctrl,
  tuneGrid = rf_grid,
  importance = "impurity",
  num.trees = 500,
  metric = "Accuracy"
)

fit_rf
## Random Forest 
## 
## 3125 samples
##   18 predictor
##    7 classes: 'Annually', 'Bi.Weekly', 'Every.3.Months', 'Fortnightly', 'Monthly', 'Quarterly', 'Weekly' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 2 times) 
## Summary of sample sizes: 2499, 2501, 2500, 2499, 2501, 2500, ... 
## Resampling results:
## 
##   Accuracy   Kappa       
##   0.1361762  -0.008473809
## 
## Tuning parameter 'mtry' was held constant at a value of 4
## Tuning parameter 'splitrule' was held constant at a value
##  of gini
## Tuning parameter 'min.node.size' was held constant at a value of 5
plot(varImp(fit_rf), top = 20, main = "RF - Variable Importance")


6.2 Model 2 — XGBoost
# building one-hot encoded matrix from training and test
mm_train <- model.matrix(frequency_of_purchases ~ . , data = train_df[, c(preds_all, "frequency_of_purchases")])
mm_test  <- model.matrix(frequency_of_purchases ~ . , data = test_df[, c(preds_all, "frequency_of_purchases")])

# extracting labels
y_train <- train_df$frequency_of_purchases
y_test  <- test_df$frequency_of_purchases

# converting to numeric indices for xgboost
label_train <- as.numeric(y_train) - 1
label_test  <- as.numeric(y_test) - 1

# preparing DMatrix
dtrain <- xgb.DMatrix(data = mm_train, label = label_train)
dtest  <- xgb.DMatrix(data = mm_test, label = label_test)

# training a simple xgboost model
set.seed(42)
fit_xgb <- xgboost(
  data = dtrain,
  objective = "multi:softmax",
  num_class = length(levels(y_train)),
  nrounds = 50,
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

# predicting on test
pred_xgb <- predict(fit_xgb, dtest)
pred_xgb <- factor(levels(y_train)[pred_xgb + 1], levels = levels(y_train))

# evaluating
cm_xgb <- caret::confusionMatrix(pred_xgb, y_test)
cm_xgb
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Annually Bi.Weekly Every.3.Months Fortnightly Monthly Quarterly Weekly
##   Annually             21        18             21          18      23        19     24
##   Bi.Weekly            12        13             14          15      14        18     17
##   Every.3.Months       27        27             22          20      16        21     17
##   Fortnightly          14         5              7          15       8         9     11
##   Monthly              12        18             13          12      15        13     12
##   Quarterly            17        20             32          18      22        19     13
##   Weekly               11         8              8          10      12        13     13
## 
## Overall Statistics
##                                           
##                Accuracy : 0.1519          
##                  95% CI : (0.1273, 0.1791)
##     No Information Rate : 0.1506          
##     P-Value [Acc > NIR] : 0.475357        
##                                           
##                   Kappa : 0.0092          
##                                           
##  Mcnemar's Test P-Value : 0.004125        
## 
## Statistics by Class:
## 
##                      Class: Annually Class: Bi.Weekly Class: Every.3.Months Class: Fortnightly Class: Monthly
## Sensitivity                  0.18421          0.11927               0.18803            0.13889        0.13636
## Specificity                  0.81448          0.86527               0.80606            0.91928        0.88006
## Pos Pred Value               0.14583          0.12621               0.14667            0.21739        0.15789
## Neg Pred Value               0.85308          0.85757               0.84848            0.86864        0.86070
## Prevalence                   0.14672          0.14028               0.15058            0.13900        0.14157
## Detection Rate               0.02703          0.01673               0.02831            0.01931        0.01931
## Detection Prevalence         0.18533          0.13256               0.19305            0.08880        0.12227
## Balanced Accuracy            0.49935          0.49227               0.49705            0.52909        0.50821
##                      Class: Quarterly Class: Weekly
## Sensitivity                   0.16964       0.12150
## Specificity                   0.81654       0.90746
## Pos Pred Value                0.13475       0.17333
## Neg Pred Value                0.85377       0.86610
## Prevalence                    0.14414       0.13771
## Detection Rate                0.02445       0.01673
## Detection Prevalence          0.18147       0.09653
## Balanced Accuracy             0.49309       0.51448
# compute feature importance
imp_mat <- xgb.importance(model = fit_xgb)

# top variables
head(imp_mat, 10)
##                                 Feature       Gain       Cover  Frequency
##                                  <char>      <num>       <num>      <num>
##  1:                           unnamed_0 0.19053722 0.214201245 0.17830128
##  2:                 purchase_amount_usd 0.14267501 0.124469889 0.13745407
##  3:                                 age 0.12735493 0.128964513 0.11843527
##  4:                  previous_purchases 0.11660730 0.118313579 0.11281608
##  5:                       review_rating 0.09438808 0.085291403 0.09920035
##  6:                         regionSouth 0.01436837 0.011502320 0.01426410
##  7: preferred_payment_methodCredit Card 0.01352810 0.012620120 0.01426410
##  8:                  payment_methodCash 0.01303452 0.022722840 0.01469635
##  9:                               sizeM 0.01178827 0.009542932 0.01296737
## 10:                          colorOther 0.01150123 0.006221307 0.01059002
# plot importance
xgb.plot.importance(imp_mat[1:15], main = "XGBoost - Top 15 Important Features")


7 Task 5 — Evaluation
# helper function for metrics
compute_metrics <- function(truth, preds) {
  cm <- caret::confusionMatrix(preds, truth)
  byclass <- cm$byClass
  if (is.matrix(byclass)) {
    f1s <- byclass[, "F1"]
    macro_f1 <- mean(f1s, na.rm = TRUE)
  } else {
    macro_f1 <- as.numeric(byclass["F1"])
  }
  list(confusion = cm$table, accuracy = cm$overall["Accuracy"], macroF1 = macro_f1)
}

# RF evaluation
pred_rf <- predict(fit_rf, newdata = test_df[, preds_all, drop = FALSE])
m_rf <- compute_metrics(test_df$frequency_of_purchases, pred_rf)

# XGB evaluation (already predicted earlier as pred_xgb)
m_xgb <- compute_metrics(y_test, pred_xgb)

m_rf
## $confusion
##                 Reference
## Prediction       Annually Bi.Weekly Every.3.Months Fortnightly Monthly Quarterly Weekly
##   Annually             18        14             21          11      28        15     18
##   Bi.Weekly            15        15             15           9       7        15     15
##   Every.3.Months       21        20             20          27      20        20     18
##   Fortnightly          10        10             12          10       2        13     12
##   Monthly              11        13             17          21      17        16     17
##   Quarterly            19        24             22          18      13        15     13
##   Weekly               20        13             10          12      23        18     14
## 
## $accuracy
##  Accuracy 
## 0.1402831 
## 
## $macroF1
## [1] 0.1392882
m_xgb
## $confusion
##                 Reference
## Prediction       Annually Bi.Weekly Every.3.Months Fortnightly Monthly Quarterly Weekly
##   Annually             21        18             21          18      23        19     24
##   Bi.Weekly            12        13             14          15      14        18     17
##   Every.3.Months       27        27             22          20      16        21     17
##   Fortnightly          14         5              7          15       8         9     11
##   Monthly              12        18             13          12      15        13     12
##   Quarterly            17        20             32          18      22        19     13
##   Weekly               11         8              8          10      12        13     13
## 
## $accuracy
##  Accuracy 
## 0.1518662 
## 
## $macroF1
## [1] 0.151302
8 Task 6 — Model improvement
# tuning Random Forest with different mtry values
rf_tune_grid <- expand.grid(mtry = c(2, 5, 10), splitrule = "gini", min.node.size = 5)

set.seed(42)
fit_rf_tuned <- train(
  x = train_df[, preds_all, drop = FALSE],
  y = train_df$frequency_of_purchases,
  method = "ranger",
  trControl = ctrl,
  tuneGrid = rf_tune_grid,
  num.trees = 500,
  metric = "Accuracy"
)

fit_rf_tuned
## Random Forest 
## 
## 3125 samples
##   18 predictor
##    7 classes: 'Annually', 'Bi.Weekly', 'Every.3.Months', 'Fortnightly', 'Monthly', 'Quarterly', 'Weekly' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 2 times) 
## Summary of sample sizes: 2499, 2501, 2500, 2499, 2501, 2500, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa       
##    2    0.1331364  -0.012129218
##    5    0.1336180  -0.011355069
##   10    0.1400098  -0.003907261
## 
## Tuning parameter 'splitrule' was held constant at a value of gini
## Tuning parameter 'min.node.size' was held constant
##  at a value of 5
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were mtry = 10, splitrule = gini and min.node.size = 5.
Random Forest tuning shows possible improvement over baseline.

XGBoost can also be tuned further (learning rate, depth, rounds).

Oversampling/weighting can address imbalance.

9 Task 7 — Comparison and conclusion
summary_tbl <- tibble(
  model = c("Random Forest", "XGBoost"),
  accuracy = c(as.numeric(m_rf$accuracy), as.numeric(m_xgb$accuracy)),
  macro_F1 = c(as.numeric(m_rf$macroF1), as.numeric(m_xgb$macroF1))
) %>% arrange(desc(accuracy))

summary_tbl
## # A tibble: 2 × 3
##   model         accuracy macro_F1
##   <chr>            <dbl>    <dbl>
## 1 XGBoost          0.152    0.151
## 2 Random Forest    0.140    0.139
# comparison plot
summary_tbl %>%
  gather(metric, value, -model) %>%
  ggplot(aes(x = model, y = value, fill = metric)) +
  geom_col(position = "dodge") +
  labs(title = "Model Comparison: Accuracy vs Macro-F1", y = "Score", x = "Model") +
  scale_fill_brewer(palette = "Set2")


10 Session Info
sessionInfo()
## R version 4.5.1 (2025-06-13 ucrt)
## Platform: x86_64-w64-mingw32/x64
## Running under: Windows 11 x64 (build 22000)
## 
## Matrix products: default
##   LAPACK version 3.12.1
## 
## locale:
## [1] LC_COLLATE=English_India.utf8  LC_CTYPE=English_India.utf8    LC_MONETARY=English_India.utf8
## [4] LC_NUMERIC=C                   LC_TIME=English_India.utf8    
## 
## time zone: Asia/Calcutta
## tzcode source: internal
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] GGally_2.4.0      ggrepel_0.9.6     vip_0.4.1         e1071_1.7-16      xgboost_1.7.11.1  ranger_0.17.0    
##  [7] caret_7.0-1       lattice_0.22-7    skimr_2.2.1       janitor_2.2.1     data.table_1.17.8 lubridate_1.9.4  
## [13] forcats_1.0.0     stringr_1.5.1     dplyr_1.1.4       purrr_1.1.0       readr_2.1.5       tidyr_1.3.1      
## [19] tibble_3.3.0      ggplot2_3.5.2     tidyverse_2.0.0  
## 
## loaded via a namespace (and not attached):
##  [1] tidyselect_1.2.1     timeDate_4041.110    farver_2.1.2         S7_0.2.0             fastmap_1.2.0       
##  [6] pROC_1.19.0.1        digest_0.6.37        rpart_4.1.24         timechange_0.3.0     lifecycle_1.0.4     
## [11] survival_3.8-3       magrittr_2.0.3       compiler_4.5.1       rlang_1.1.6          sass_0.4.10         
## [16] tools_4.5.1          utf8_1.2.6           yaml_2.3.10          knitr_1.50           labeling_0.4.3      
## [21] plyr_1.8.9           repr_1.1.7           RColorBrewer_1.1-3   withr_3.0.2          nnet_7.3-20         
## [26] grid_4.5.1           stats4_4.5.1         future_1.67.0        globals_0.18.0       scales_1.4.0        
## [31] iterators_1.0.14     MASS_7.3-65          cli_3.6.5            crayon_1.5.3         rmarkdown_2.29      
## [36] generics_0.1.4       rstudioapi_0.17.1    future.apply_1.20.0  reshape2_1.4.4       tzdb_0.5.0          
## [41] cachem_1.1.0         proxy_0.4-27         splines_4.5.1        parallel_4.5.1       base64enc_0.1-3     
## [46] vctrs_0.6.5          hardhat_1.4.2        Matrix_1.7-3         jsonlite_2.0.0       hms_1.1.3           
## [51] listenv_0.9.1        foreach_1.5.2        gower_1.0.2          jquerylib_0.1.4      recipes_1.3.1       
## [56] glue_1.8.0           parallelly_1.45.1    ggstats_0.10.0       codetools_0.2-20     stringi_1.8.7       
## [61] gtable_0.3.6         pillar_1.11.0        htmltools_0.5.8.1    ipred_0.9-15         lava_1.8.1          
## [66] R6_2.6.1             evaluate_1.0.5       snakecase_0.11.1     bslib_0.9.0          class_7.3-23        
## [71] Rcpp_1.1.0           nlme_3.1-168         prodlim_2025.04.28   xfun_0.52            pkgconfig_2.0.3     
## [76] ModelMetrics_1.2.2.2
