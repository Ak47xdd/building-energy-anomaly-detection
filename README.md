# building-energy-anomaly-detection

A repository for analyzing and detecting anomalies in building energy consumption data. The project contains data, analysis notebooks, visualizations, and tools for performing exploratory data analysis and building anomaly detection models.

# Table of Contents

• About the Project

• Repository Structure

• Dataset

• Requirements

• Installation

• Usage

• Notebooks Overview

• Example Workflow

• Plotting and Outputs

1. About the Project
   
This project aims to analyze building energy meter data and detect abnormal consumption patterns (anomalies). Detecting energy anomalies can be crucial for identifying inefficiencies, equipment failures, or unusual usage patterns, which can help building managers decrease energy waste.

2. Repository Structure
   
Raw energy meter time-series data
Jupyter notebooks for analysis and modeling Visualization plots and
CSV files (Data)

3. Dataset
   
Raw energy meter data resides under data/meters/raw. These are time-series files containing timestamps and energy measurements for various buildings or meters.

Typical files contain:

• Timestamp (date/time of measurement)

• Energy consumption reading (e.g., kWh)

• Meter identifiers (if multiple meters)

Data is typically used for:

• Exploratory data analysis

• Training anomaly detection models

• Evaluating detection performance against expected patterns

5. Requirements
   
Python packages required for this project are listed in requirements.txt.

Typical dependencies include: 

```code
Pandas
Numpy
Scikitlearn
Seaborn
joblib
os
```
5. Installation

Clone the repository

```code
git clone https://github.com/Fuhrerx/building-energy-anomaly-detection.git
cd building-energy-anomaly-detection
```

Install dependencies, It is recommended to use a virtual environment:

```code
python -m venv env
source env/bin/activate         # Linux / macOS
.\env\Scripts\activate          # Windows
pip install -r requirements.txt
```

6. Usage
   
Running Notebooks
Most analysis is done via Jupyter notebooks. 

To start the notebook server:

``` code
jupyter notebook
```

Open and run the notebooks in VSCode/ Jupyter or Colab to preprocess data, visualize metrics, and train or evaluate models.

7. Notebooks Overview
   
Each notebook serves a specific purpose. The recommended sequence is : 

• Explotory Data Analysis (01_eda.ipynb)

• Data Preprocessing (02_preprocessing.ipynb)

• Feature Engineering (03_feature_engineering.ipynb)

• Model Building (04_model_building.ipynb)

9. Example Workflow
    
• Load and inspect raw data.

• Preprocess data (e.g., filling missing values, normalization).

• Visualize metrics to understand trends and seasonal effects.

• Train anomaly detection models (e.g., threshold methods, clustering, ML).

• Evaluate detected anomalies and possibly visualize results.

11. Plotting and Outputs
    
Visualization plots generated from analysis are stored in the plots/ directory. 

Typical plots include:

• Time series of energy consumption

• Detected anomaly markers

• Model performance charts
