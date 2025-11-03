# An Interpretable Machine Learning Pipeline for Day-Ahead Rain/No-Rain Prediction in Basel

This project uses the Kaggle “Weather Prediction” dataset (derived from the European Climate Assessment & Dataset, ECA&D, 2000–2010, 18 European stations) to answer a single supervised question: **can I predict “Rain vs No Rain tomorrow” for Basel better than trivial rules like “always no rain” or “tomorrow = today”?** The motivation is to build a clean, teaching-style ML pipeline that is transparent, explainable, and easy to grade: I load the full multi-city weather file, focus on Basel variables, create the label by shifting Basel precipitation by one day, explore the data with several visualizations, and then train baselines (majority, simple logistic regression) to establish what performance is actually achievable from station data alone before moving on to more feature-rich models. This is not meant to be a black-box weather app; it is meant to show good ML practice on real data (EDA → target definition → baseline → improvement).

## Directory Structure

```text
cmse492_project/
├── README.md                          # Project overview, setup, usage (this file)
├── requirements.txt                   # Python dependencies for the project (didn't needed so far)
├── .gitignore                         # Ignore data, local env, checkpoints, etc.
├── data/
│   └── raw/
│       └── weather_prediction.csv     # Kaggle/ECA&D CSV (not in git)
├── notebooks/
│   └── exploratory/
│       ├── B1_data_loading.ipynb      # load + document source
│       ├── B2_initial_eda.ipynb       # 4+ visualizations, saved to figures/
│       └── B3_baseline_model.ipynb    # majority + logistic regression baseline
├── figures/                           # all saved plots from B2 (dist, scatter, month, drivers…)
└── src/                               # (optional) helper code / feature builders
```

I am keeping the raw data under data/raw/ because that is what I load in every notebook. I do not currently store processed data because most cleaning/feature work is happening directly in the notebooks, but the folder structure leaves room for data/processed/ later if I export a Basel-only subset. All EDA plots (distribution, humidity vs temperature, rain class balance, rain by month, drivers vs precipitation) are saved in figures/ with descriptive names so they can be included in the report.

## Setup Instructions

To run this project, I first make sure I have Python 3.9+ and a virtual environment (optional but recommended). Then I install the few packages I actually use: pandas (for loading and cleaning the CSV), numpy (numerics), matplotlib (to make and save the EDA figures in figures/), and scikit-learn (to train the baseline models in B3, e.g. logistic regression and to compute accuracy/F1/confusion matrix). This can be done with one command like:

pip install pandas numpy matplotlib scikit-learn

After that, I download the Kaggle/ECA&D weather file and place it at data/raw/weather_prediction.csv so that all notebooks can read it with the same path. Then I open the notebooks in order: B1_data_loading.ipynb to show the dataset shape and dtypes, B2_initial_eda.ipynb to create the four required plots (distribution, relationship, class balance, seasonality) and save them to figures/, and B3_baseline_model.ipynb to build the RainTomorrow label and evaluate the two baselines. No separate requirements.txt is strictly needed because the dependencies are listed here and can be installed directly.
