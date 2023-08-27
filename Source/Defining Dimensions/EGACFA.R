# load the necessary packages
library(lavaan, quietly=TRUE, warn.conflicts=FALSE)
library(readr, quietly=TRUE, warn.conflicts=FALSE)  # Data import
library(psych, quietly=TRUE, warn.conflicts=FALSE)
print(getwd())
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)
library(purrr, quietly=TRUE, warn.conflicts=FALSE)

print(getwd())
source("Source/DataPreparation/functions.R")

# Importing the two datasets
train_df <-read_csv("data\\processed\\factordatasets\\traindf.csv", show_col_types = FALSE)
test_df <-read_csv("data\\processed\\factordatasets\\testdf.csv", show_col_types = FALSE)

# Importing the dimensions
EGAdimensions <-read_csv2("data\\EGAdimensions.csv", show_col_types = FALSE)


# Removing the variables to remove numerical and dichotomic
cfatrain_df <- train_df %>% select(EGAdimensions$names,eng_energy,eng_enthusiastic,eng_timeflies)
cfatest_df <- test_df %>% select(EGAdimensions$names,eng_energy,eng_enthusiastic,eng_timeflies)

# Creating the string for the model
model_string <- map_chr(unique(EGAdimensions$EGA_dimension), function(dimension) {
  factor_variables <- EGAdimensions$names[EGAdimensions$EGA_dimension == dimension]
  
  # If there are only two observed variables, fix lambda
  if(length(factor_variables) == 2) {
    return(paste(dimension, " =~ lambda*", factor_variables[1], "+ lambda*", factor_variables[2], ";"))
  } 
  
  # Otherwise, return the regular line
  paste(dimension, " =~ ", paste(factor_variables, collapse = " + "), ";")
})

# Append the additional line to model_string
model_string <- paste(c(model_string, "Work Engagement =~ eng_energy + eng_enthusiastic + eng_timeflies;"), collapse = " ")
#cat(paste(strsplit(model_string, ";")[[1]], sep = "\n"), sep = "\n")

print("Starting CFA")

# Fit the CFA model to the training and testing data
fittrain <- cfa(model_string, data=cfatrain_df, estimator="WLSMV", missing = "pairwise",ordered=TRUE)
fittest <- cfa(model_string, data=cfatest_df, estimator="WLSMV", missing = "pairwise",ordered=TRUE)
summary(fittrain, fit.measures = TRUE, standardized = T)
#summary(fittest, fit.measures = TRUE, standardized = T)


# Compute factor scores for the training and testing data
factor_scores_train <- lavPredict(fittrain, type="lv", method="ebm")
factor_scores_test <- lavPredict(fittest, type="lv", method="ebm")


# Update 'train_df' and 'test_df' by removing the old variables, keeping a few specific ones,
# and adding in the new factor scores
train_df <- train_df %>%
  select(-all_of(EGAdimensions$names)) %>%
  bind_cols(factor_scores_train)

test_df <- test_df %>%
  select(-all_of(EGAdimensions$names)) %>%
  bind_cols(factor_scores_test)

print("Saving Updated datasets")
# Write the updated 'train_df' and 'test_df' back out to .csv files
write_csv(train_df, "data\\processed\\factordatasets\\cfatrain.csv")
write_csv(test_df, "data\\processed\\factordatasets\\cfatest.csv")

print("Done")
