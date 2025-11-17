# An Interpretable Machine Learning Pipeline for Day-Ahead Rain/No-Rain Prediction in Basel

This project uses the Kaggle “Weather Prediction” dataset, derived from the European Climate Assessment & Dataset (ECA&D) covering the years 2000–2010 for 18 European weather stations, to answer one central question: **Can we predict “Rain vs No Rain tomorrow” for Basel better than simple rules such as “always no rain” or “tomorrow = today”?** Rather than building a complex black-box system, the aim is to construct a clean, teaching-focused machine learning pipeline that is easy to follow, transparent in its assumptions, and straightforward to grade. The workflow begins by loading the full multi-city weather file, then filtering the data to Basel, and defining a day-ahead target variable by shifting Basel’s precipitation by one day. From there, the project performs weather-sensible exploratory data analysis, establishes baseline models, and gradually introduces more expressive models while keeping interpretation at the center.

The project emphasizes good, interpretable ML practice on real data. This includes clearly documenting the raw dataset, conducting exploratory data analysis that respects meteorological structure (such as temperature distributions, humidity–temperature relationships, class imbalance between rainy and non-rainy days, and seasonal patterns in rainfall), defining a time-aware binary target RainTomorrow, and starting with simple baselines like a majority-class rule, “tomorrow = today” behavior, and a basic logistic regression. Only after these baselines are understood does the pipeline move on to more feature-rich models, such as tuned logistic regression, random forests, and a small neural network, and then interprets what each model appears to be learning through coefficients, feature importance, and calibration analysis.

### Problem Defination

The input to the problem consists of daily weather measurements for Basel extracted from a larger multi-station European dataset. These features include standard meteorological variables such as temperature, humidity, and potentially additional drivers that influence precipitation. The target variable, RainTomorrow, is defined by shifting the recorded precipitation for Basel one day into the future so that each row’s features describe “today” while the label indicates whether it will rain “tomorrow.” This turns the task into a supervised binary classification problem with two classes: “Rain” and “No Rain.”

Conceptually, the project compares naive weather rules against learned models. Naive baselines include always predicting the majority class (“always no rain”) and using the simple heuristic that tomorrow’s rain status will match today’s. Learned models include logistic regression, random forest classifiers, and a small multilayer perceptron (MLP). The main scientific and educational question is how much improvement an interpretable model can achieve over trivial rules when it is trained on structured station data.

---

### Directory Structure
The project is organized into a standard directory layout that separates raw data, processed data, figures, notebooks, and reusable code:
```text
cmse492_project/
├── README.md                      # Project overview, setup, usage (this file)
├── requirements.txt               # Python dependencies (not required , empty)
├── .gitignore                     # Ignore data, local env, checkpoints, etc.
├── data/
│   ├── raw/
│   │   └── weather_prediction_dataset.csv   # Kaggle/ECA&D CSV 
│   └── processed/
│       └── basel_rain_features.csv          # Basel-only features + labels
|       └── improved_model_results.csv
├── figures/                       # All saved plots (EDA + evaluation)
├── notebooks/
│   ├── exploratory/
│   │   ├── B1_data_loading.ipynb          # load + document source
│   │   └── B2_initial_eda.ipynb           # temperature dist, humidity vs temp,
│   │                                      # rain class balance, rain by month, drivers vs precip
│   ├── preprocessing/
│   │   └── preprocessing.ipynb         # build Basel-only feature table and RainTomorrow label
│   ├── Modeling/
│   │   ├── baseline.ipynb                 # majority baseline + simple logistic regression
│   │   ├── Improved_model.ipynb           # tuned logistic regression + random forest
│   │   └── neural_network.ipynb        # MLP neural network baseline and comparison
│   │   └── example_prediction.ipynb`    # example how to use our model 
│   └── evaluation/
│       ├── model_interpretation.ipynb  # coefficients, thresholds, feature importances
│       └── Additional_plots.ipynb         # probability histograms, calibration, monthly recall, etc.
└── src/                                   # reusable helper code
    ├── preprocessing/
    │   └── features.py                    # load_raw_weather, make_basel_features
    ├── models/
    │   └── baseline.py                    # make_logistic_pipeline, make_random_forest, make_mlp_classifier
    └── evaluation/
        └── metrics.py                     # eval_model, (optionally threshold_metrics)
```
This structure keeps the raw data immutable in data/raw, ensures that processed tables live in data/processed, and allows all experiments to be run via notebooks that share a common set of helper functions from src/.

### Methods and Modeling Approach
The workflow proceeds through several stages: exploratory data analysis, preprocessing, baseline modeling, improved modeling, and evaluation with interpretation.

During exploratory data analysis, the notebooks first confirm that the dataset loads correctly and inspect shapes, column names, data types, and missing values. They then examine the empirical distributions of key variables such as daily temperatures, relationships between variables like humidity and temperature, the proportion of days labeled as rainy versus non-rainy, and how rainfall patterns change by month or season. These visualizations are saved in the figures/ directory for later use in the report.

In the preprocessing stage, the project filters the multi-city dataset down to Basel alone and constructs a clean feature table for that location. Using helper functions defined in src/preprocessing/features.py, it creates the RainTomorrow label by shifting precipitation one day ahead, ensuring that the predictors correspond to “day t” while the label corresponds to “day t + 1.” Additional physics- or weather-guided features can be engineered at this stage if desired. The resulting processed table is stored as data/processed/basel_rain_features.csv so that multiple notebooks can use the same consistent dataset.

The baseline modeling stage focuses on extremely simple predictors. One baseline always predicts the most common class in the training data, providing a trivial reference for accuracy and F1 score. Another baseline uses a straightforward logistic regression model with basic preprocessing, such as standardization, and measures performance using metrics like accuracy, precision, recall, F1 score, and confusion matrices. These models help answer whether there is enough signal in the station data to justify more complex approaches.

The improved models build on this foundation by tuning the logistic regression (for example, via regularization strength or class weights), and adding a random forest classifier that can capture nonlinear interactions between variables. A small multilayer perceptron (MLP) neural network is also trained as a simple neural baseline. The goal is not to maximize predictive performance at all costs, but to explore how modestly more expressive models behave relative to the interpretable baselines.

Finally, the evaluation and interpretation stage examines the models in more depth. Using helper functions from src/evaluation/metrics.py and dedicated notebooks in the evaluation/ directory, the project generates confusion matrices, calculates F1 scores, and, if implemented, draws ROC curves and computes AUC values. It also interprets logistic regression coefficients to see which variables are associated with increased or decreased rain probability, analyzes feature importances from the random forest, and inspects probability calibration and decision thresholds. Additional plots explore how performance varies by month or season and how well each model separates rainy and non-rainy days in terms of predicted probabilities.

### Results

After running the preprocessing, modeling, and evaluation notebooks, I can summarize the performance of the main models on the held-out test period (2008–2010). The majority baseline, which always predicts “No Rain,” gets roughly half of the days correct simply because dry days are slightly more common, but it completely fails to identify rainy days and has an F1 score of 0 for the rain class. A simple logistic regression using a small set of Basel features (pressure, humidity, mean temperature, sunshine) already improves on this trivial rule and shows that the station-level weather variables contain real predictive signal. The tuned logistic regression, trained on the engineered Basel feature table (including month and one-day lags) with scaling and class weighting, reaches about 0.668 test accuracy and an F1 score of 0.668 for the rain class, with precision around 0.64 and recall around 0.70. A random forest trained on the same features performs very similarly, with accuracy near 0.67 and F1 for rainy days around 0.65–0.66, and its feature importance rankings largely agree with the logistic regression about which variables matter most (pressure and its lag, temperature and its lag, sunshine, and recent rain). A small neural network (MLP), tuned with a modest grid search over hidden-layer size and regularization strength, reaches roughly 0.680 accuracy and an F1 score around 0.675 for rainy days. This is only a small improvement over the tuned logistic regression, on the order of one or two percentage points, which supports the project’s main point: for this single-station, day-ahead rain/no-rain task, a clean, physics-guided logistic regression model performs almost as well as more complex models while remaining much easier to interpret and explain.

### Limitations and Future Directions

This project is intentionally focused on interpretability and clarity rather than on building the most powerful possible forecasting system, so several limitations are built in by design. The analysis is restricted to a single station (Basel) and to a fixed historical period (2000–2010), which may limit how well the results generalize to other locations, climates, or more recent years. The models work with daily aggregated data and a relatively small set of basic features, without using higher-frequency measurements or more sophisticated derived variables such as pressure trends or multi-day weather patterns. Methodologically, the project uses straightforward supervised learning models—logistic regression, random forest, and a small MLP—and does not explore more advanced time-series or spatial approaches, such as sequence models or systems that jointly model multiple stations. Hyperparameter tuning is also intentionally modest, focusing on a few sensible settings rather than a large automated search.

These choices make sense for a course project that emphasizes an interpretable, well-documented pipeline, but they leave several natural directions for future work. The project could be extended by incorporating more stations and testing how models trained on Basel transfer to other cities, or by training multi-station models that use spatial information. Richer weather features could be added to better capture dynamics, such as day-to-day pressure changes, wind-related variables, or more detailed radiation and cloud-cover measures. More advanced models, including deeper neural networks, gradient-boosted trees, or sequence-based architectures, could be tested, and more careful handling of missing data and calibration could be explored. These directions would move the work closer to a practical forecasting system while building on the interpretable baseline developed here.

### Acknowledgements

The dataset used in this project is taken from Kaggle’s “Weather Prediction” data, which itself is derived from the European Climate Assessment & Dataset (ECA&D). This project was developed as part of a CMSE course to practice building an end-to-end machine learning pipeline on real-world data, with a strong emphasis on interpretability, clear documentation, and reproducible structure. I am grateful for the publicly available ECA&D data and for the course framework that encouraged organizing the work into a clean directory structure, well-labeled notebooks, and small reusable helper modules in `src/`, which together made the project easier to understand, extend, and grade.


### Setup Instructions

To run this project, I first make sure I have Python 3.9+ and a virtual environment (optional but recommended). Then I install the few packages I actually use: pandas (for loading and cleaning the CSV), numpy (numerics), matplotlib (to make and save the EDA figures in figures/), and scikit-learn (to train the baseline models in B3, e.g. logistic regression and to compute accuracy/F1/confusion matrix). This can be done with one command like:

pip install pandas numpy matplotlib scikit-learn


After that, I download the Kaggle/ECA&D weather file and place it at data/raw/weather_prediction_dataset.csv so that all notebooks can read it with the same path. From the top-level cmse492_project folder, I start Jupyter with jupyter notebook, then open the notebooks in order: notebooks/exploratory/B1_data_loading.ipynb to show the dataset shape and dtypes, notebooks/exploratory/B2_initial_eda.ipynb to create the main plots (distribution, relationships, class balance, seasonality) and save them to figures/, notebooks/preprocessing/preprocessing.ipynb to build the Basel-only feature table and the RainTomorrow label and save it to data/processed/basel_rain_features.csv, notebooks/Modeling/baseline.ipynb and notebooks/Modeling/Improved_model.ipynb to train and evaluate the baseline and improved models, notebooks/evaluation/model_interpretation.ipynb and notebooks/evaluation/Additional_plots.ipynb to interpret the models and generate extra diagnostics, and finally notebooks/Modeling/neural_network.ipynb to train and compare the neural network. No separate requirements.txt is strictly needed because the few dependencies are listed here and can be installed directly.