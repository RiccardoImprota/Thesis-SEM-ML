# +-----------------------------------------+
# |          R Libraries Import             |
# +-----------------------------------------+

# Tidyverse Libraries for Data Wrangling & Visualization
library(dplyr)      # Data Manipulation
library(purrr)      # Functional Programming Tools
library(readr)      # Data Input & Output
library(tidyr)      # Tidy Data



# ***********************************************************************
# ***********************************************************************

# +-----------------------------------------+
# |                                         |
# |           Data Parsing Section          |
# |                                         |
# +-----------------------------------------+


# This function imports a list of variables that might be used,
# and filters them based on whether we want questions in the M1 part of the
# survey or not
# 
# Params:
# - M1: A boolean value indicating whether to consider M1 questions or not.
#       Default is TRUE.
#
# Returns: A filtered dataframe of used variables.

ImportUsedvar <- function( M1 = TRUE) {
  usedvar <- read_csv2("../data/used_variables.csv", show_col_types = FALSE)
  
  if (!M1) {
    usedvar <- usedvar %>% 
      filter(is.na(M1) | M1 != 1)
  }
  
  return(usedvar)
}

# This function imports the original dataset stored in SPSS format.
#
# Returns: The original dataset.

ImportOriginalDataset <- function(){
  sourcedf <- read_spss("../data/source/ewcts_2021_isco2_nace2_nuts2.sav") 
  return(sourcedf)
}

# This function imports the reversed dataframe processed by the data_reverse.qmd script
#
# Returns: The dataframe where the variables have been reversed.

ImportReversedDF <- function(){
  sourcedf <- read_csv("../data/processed/df_reversed.csv",show_col_types = FALSE) 
  return(sourcedf)
}

