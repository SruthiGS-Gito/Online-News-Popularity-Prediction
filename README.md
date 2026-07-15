# Online News Popularity Prediction

## Project Overview
This project predicts the number of shares an online news article will get, using the UCI Online News Popularity dataset. The goal is to preprocess the data and prepare it for model building.

## Dataset Source
UCI Machine Learning Repository - Online News Popularity Dataset
https://archive.ics.uci.edu/dataset/332/online+news+popularity

## Preprocessing Pipeline
1. **EDA** - Checked shape, data types, missing values, duplicates, and column distributions using histograms and boxplots.
2. **Dropped Non-Predictive Columns** - Removed `url` (identifier, not a feature) and `timedelta` (metadata, not predictive).
3. **Correlation & Mutual Information** - Computed Pearson correlation and mutual information scores against `shares` to select the most relevant features (`key_cols`).
4. **Outlier Handling** - Applied IQR-based capping on key numerical columns to reduce the effect of extreme values.
5. **Data Split** - Split the data into train and test sets before scaling, to avoid data leakage.
6. **Scaling** - Applied StandardScaler, fit only on the training set and used to transform both train and test sets.
7. **Encoding** - Not required; no categorical columns remained after dropping `url`.

## How to Run
1. Open `Online_News_Popularity_Prediction.ipynb` in Google Colab.
2. Mount Google Drive and update the dataset filepath if needed.
3. Run all cells in order from top to bottom.

## Team Members
- [Arshak Nihal] - EDA, Preprocessing setup
- [Layana] - Outlier Handling
- [Jyothish M] - Heatmap, README
- [Sruthi G S] - Data Split, Scaling, Correlation & Mutual Information
