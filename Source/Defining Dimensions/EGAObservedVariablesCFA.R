# load the necessary packages
suppressPackageStartupMessages(library(lavaan, quietly=TRUE, warn.conflicts=FALSE))
library(readr, quietly=TRUE, warn.conflicts=FALSE)  # Data import
library(psych, quietly=TRUE, warn.conflicts=FALSE)
print(getwd())
library(dplyr, quietly=TRUE, warn.conflicts=FALSE)
library(purrr, quietly=TRUE, warn.conflicts=FALSE)

print(getwd())
source("Source/DataPreparation/functions.R")

# Importing the two datasets
completedf <-read_csv("data\\processed\\df_reversed.csv", show_col_types = FALSE)
EGAdimensions <-suppressMessages(read_csv2("data\\EGAdimensions.csv", show_col_types = FALSE))

# Removing the variables to remove numerical and dichotomic
cfacompletedf <- completedf %>% select(EGAdimensions$names,eng_energy,eng_enthusiastic,eng_timeflies)


model_string <- "Work Engagement =~ eng_energy + eng_enthusiastic + eng_timeflies;"

fit <- cfa(model_string, data=cfacompletedf, estimator="WLSMV", missing = "pairwise",ordered=TRUE)
summary(fit, fit.measures = TRUE, standardized = T)

# Compute factor scores
factor_scores <- lavPredict(fit, type="lv", method="ebm")

# Update 'completedf' by removing the old variables, keeping a few specific ones,
# and adding in the new factor scores
finaldf <- completedf %>%
  select(-c(eng_energy, eng_enthusiastic, eng_timeflies)) %>%
  bind_cols(factor_scores)

print("Saving Updated datasets")
# Write the updated 'train_df' and 'test_df' back out to .csv files
write_csv(finaldf, "data\\processed\\factordatasets\\CFAObservedf.csv")

print("Done")

