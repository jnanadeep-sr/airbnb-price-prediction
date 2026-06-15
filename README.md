# Airbnb Price Prediction Project

This repository contains the final project for the SoSe 2026 Machine Learning course. The goal of the project is to build a robust, multimodal machine learning pipeline to predict nightly listing prices for Airbnb accommodations using data from the Inside Airbnb open-data initiative.

## 📊 Data Setup Instructions

Because the raw Airbnb dataset is quite large, it is ignored by Git and will not be pushed to GitHub. To run the project notebooks and scripts locally, follow these steps to manually set up your directories:

1. **Download the Dataset:**
   * Obtain the `listings.csv` and `calendar.csv.gz` datasets (e.g., from Inside Airbnb).

2. **Place the Files in the Repository:**
   * Navigate to the project root directory.
   * Drop the raw files directly into the `data/raw/` subdirectory.

Your local directory structure must match this layout exactly for the pipeline scripts to successfully locate the data:

```text
airbnb-price-prediction/
└── data/
    ├── processed/
    └── raw/
        ├── calendar.csv.gz
        └── listings.csv