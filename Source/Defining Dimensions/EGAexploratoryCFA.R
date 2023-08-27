# This code is not necessary to run any script.
# It is only used to compute the cronbach alpha of the EGA dimensions.

# load the necessary packages
library(lavaan)
library(readr)  # Data import
library(psych)
print(getwd())

source("../DataPreparation/functions.R")

df <-ImportReversedDF()

EGAdimensions <-read_csv2("../../data/EGAdimensions.csv", show_col_types = FALSE)

# Select the variables
df <- df %>%
  select(all_of(EGAdimensions$names))

model_list <- list()
for(dimension in unique(EGAdimensions$EGA_dimension)){
  factor_variables <- EGAdimensions$names[EGAdimensions$EGA_dimension == dimension]
  model_list[[dimension]] <- paste(dimension, " =~ ", paste(factor_variables, collapse = " + "), ";")
}

for(dimension in unique(EGAdimensions$EGA_dimension)){
  factor_variables <- EGAdimensions$names[EGAdimensions$EGA_dimension == dimension]
  if(length(factor_variables) == 2) {
    # If there are only two observed variables, fix lambda
    model_list[[dimension]] <- paste(dimension, " =~ lambda*", factor_variables[1], "+ lambda*", factor_variables[2], ";")
  } else {
    model_list[[dimension]] <- paste(dimension, " =~ ", paste(factor_variables, collapse = " + "), ";")
  }
}


model_string <- paste(model_list, collapse = " ")

cat(paste(model_list, collapse = "\n"))

EGAdimensions$EGA_dimension %>% 
  unique() %>% 
  purrr::map_df(~list( dimension = .x,
                       alpha = psych::alpha(df[, EGAdimensions$names[EGAdimensions$EGA_dimension == .x]],check.keys = TRUE)$total[["std.alpha"]]
  ))


# Fit the model
fit <- cfa(model_string, data=df, estimator="WLSMV", missing = "pairwise",ordered=TRUE)

summary(fit, fit.measures = TRUE, standardized = T)

