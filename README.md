\# 🌾 Crop Production Prediction Based on Agricultural Data



\## Project Overview

A machine learning project that predicts crop production (in tonnes) based on agricultural factors like area harvested, yield, crop type, country, and year. Built using FAOSTAT data (2019-2023) covering 200 countries and 157 crops.



\## Tech Stack

\- Python, Pandas, NumPy, Scikit-learn

\- Matplotlib, Seaborn, Plotly

\- Streamlit (Interactive Dashboard)

\- Jupyter Notebook



\## Dataset

\- Source: FAOSTAT (Food and Agriculture Organization)

\- Period: 2019-2023

\- Scope: 200 countries, 157 crops

\- Records: 44,827 (after cleaning)



\## Models Trained

| Model | R² (Test) | MAE |

|-------|-----------|-----|

| Linear Regression | 0.34 | 1,342,379 |

| Decision Tree | 0.96 | 88,965 |

| Random Forest | 0.97 | 84,421 |

| Gradient Boosting | 0.99 | 241,113 |

| Random Forest (Tuned) | 0.98 | 81,501 |



\## Best Model: Random Forest (Tuned)

\- R² Score: 0.9752

\- MAE: 81,501 tonnes

\- Hyperparameters: max\_depth=25, n\_estimators=300, min\_samples\_split=2



\## Project Structure

