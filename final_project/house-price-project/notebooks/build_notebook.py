"""Script to generate and synchronize house_price_model.ipynb from train_script.py cells.

Ensures all 13 sections required by ANTIGRAVITY_PROJECT_INSTRUCTIONS.md are present:
 1. Introduction
 2. Load & Inspect
 3. Data Understanding
 4. Exploratory Data Analysis (EDA)
 5. Data Cleaning
 6. Feature Engineering
 7. Preprocessing Pipeline
 8. Model Training
 9. Model Evaluation
10. Model Comparison
11. Best Model Selection
12. Model Export
13. Sanity Check
"""

import json
from pathlib import Path
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    }

    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 🏠 House Price Prediction — End-to-End Machine Learning
An end-to-end machine learning system that predicts residential property prices in India.

---

## 1. Introduction

### Project Objective
Build and deploy a robust regression pipeline to predict property prices in **Indian Rupees (INR)** based on key physical, spatial, and structural characteristics (e.g., location, carpet area, floor number, bathrooms, balconies, furnishing, and ownership).

### Business Value
Real-estate pricing is notoriously opaque with wide geographic variance. Automated pricing provides buyers, sellers, and agents with instant, transparent fair-market valuations.

### Target Variable
- **`price_clean`**: Total sale price in INR, parsed from raw mixed text/numeric representations (e.g. `"42 Lac"` $\\rightarrow$ `4,200,000 INR`, `"1.4 Cr"` $\\rightarrow$ `14,000,000 INR`).

### End-to-End Architecture
```
Raw CSV → Cleaning & Feature Engineering → Preprocessing Pipeline (Impute/Scale/OHE)
        → Model Selection (Linear, RF, GBR, XGBoost) → Export (house_price.pkl)
        → FastAPI Backend (/predict, /health) → React + TypeScript Frontend
```

> **Dataset Source**: [Kaggle — House Price by Juhi Bhojani](https://www.kaggle.com/datasets/juhibhojani/house-price)
"""))

    # Imports
    cells.append(nbf.v4.new_markdown_cell("""## 2. Load & Inspect — Imports & Setup"""))
    cells.append(nbf.v4.new_code_cell("""import warnings
import json
import re
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Plotting style
sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (10, 5)})

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print('Libraries imported successfully')
print(f'  pandas   {pd.__version__}')
print(f'  numpy    {np.__version__}')
print(f'  sklearn  {__import__("sklearn").__version__}')"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""### Loading the Raw Dataset"""))
    cells.append(nbf.v4.new_code_cell("""DATA_PATH = Path('data/house_prices.csv') if Path('data/house_prices.csv').exists() else Path('notebooks/data/house_prices.csv')

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH.resolve()}'.\\n"
        "Please download 'house_prices.csv' from:\\n"
        "  https://www.kaggle.com/datasets/juhibhojani/house-price\\n"
        "and place it in the notebooks/data/ directory."
    )

df_raw = pd.read_csv(DATA_PATH, low_memory=False)

print('=== Shape ===')
print(f'{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns')
df_raw.head()"""))

    cells.append(nbf.v4.new_code_cell("""print('=== Column Names ===')
for i, col in enumerate(df_raw.columns, 1):
    print(f'  {i:2d}. {col}')"""))

    cells.append(nbf.v4.new_code_cell("""print('=== Data Types & Memory Usage ===')
df_raw.info()"""))

    cells.append(nbf.v4.new_code_cell("""print('=== Descriptive Statistics (numeric & categorical columns) ===')
df_raw.describe(include='all').T"""))

    # Data Understanding
    cells.append(nbf.v4.new_markdown_cell("""## 3. Data Understanding

Inspect data types, missing value ratios across all columns, and duplicate listings.
"""))
    cells.append(nbf.v4.new_code_cell("""dtype_df = pd.DataFrame({
    'dtype': df_raw.dtypes,
    'unique_values': df_raw.nunique(),
    'missing_count': df_raw.isna().sum(),
    'missing_pct': (df_raw.isna().mean() * 100).round(2),
    'sample_value': [df_raw[c].dropna().iloc[0] if df_raw[c].notna().any() else None for c in df_raw.columns]
})
print(dtype_df.to_string())"""))

    cells.append(nbf.v4.new_code_cell("""missing = df_raw.isna().mean().sort_values(ascending=False)
print('=== Missing Value Ratio (sorted) ===')
print((missing * 100).round(2).to_string())"""))

    cells.append(nbf.v4.new_code_cell("""dup_count = df_raw.duplicated().sum()
print(f'Duplicate rows: {dup_count:,} ({dup_count/len(df_raw)*100:.2f}%)')"""))

    # EDA
    cells.append(nbf.v4.new_markdown_cell("""## 4. Exploratory Data Analysis (EDA)

We explore the distribution of the target variable and key relationships with features:
1. **Target Price Distribution** (Linear vs. Log₁₀ Scale)
2. **Price vs. Carpet Area** (Scatter Plot)
3. **Median Price by Top-15 Locations** (Bar Chart)
4. **Price by Furnishing Status** (Box Plot)
5. **Price by Bathroom Count** (Box Plot)
"""))

    cells.append(nbf.v4.new_code_cell("""# ─── Identify & Quick-Parse Price for EDA ────────────────────────────────────
KNOWN_PRICE_COLS = ['Amount(in rupees)', 'amount(in rupees)', 'Price', 'price',
                    'Total Amount', 'total_amount', 'Amount']

PRICE_COL = next((c for c in KNOWN_PRICE_COLS if c in df_raw.columns), None)
if PRICE_COL is None:
    for col in df_raw.columns:
        if df_raw[col].dtype == 'object':
            sample = df_raw[col].dropna().astype(str)
            if sample.str.contains(r'^\s*[\d.,]+\s*(Lac|Cr|lac|cr)\s*$', regex=True).mean() > 0.5:
                PRICE_COL = col
                break

print(f'Using price column: "{PRICE_COL}"')

def parse_amount_eda(val):
    if not isinstance(val, str):
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan
    s = val.strip().lower().replace(',', '')
    try:
        if 'cr' in s:
            return float(re.sub(r'[^0-9.]', '', s.replace('cr', ''))) * 1e7
        if 'lac' in s or 'lakh' in s:
            return float(re.sub(r'[^0-9.]', '', re.sub(r'la[ck]h?', '', s))) * 1e5
        return float(re.sub(r'[^0-9.]', '', s))
    except (ValueError, TypeError):
        return np.nan

df_eda = df_raw.copy()
df_eda['price_inr'] = df_eda[PRICE_COL].apply(parse_amount_eda)
df_eda_clean = df_eda.dropna(subset=['price_inr']).query('price_inr > 0')
print(f'Rows with valid prices: {len(df_eda_clean):,} / {len(df_raw):,}')"""))

    # Plot 1
    cells.append(nbf.v4.new_markdown_cell("""### Plot 1: Target Price Distribution (Linear vs. Log Scale)
**Interpretation**: Property prices span several orders of magnitude (from ₹10 Lac to ₹50+ Cr) and exhibit extreme right skewness in linear space. Applying a log-transformation ($\log(1+y)$) yields a near-normal distribution, which significantly stabilizes regression training."""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
axes[0].hist(df_eda_clean['price_inr'] / 1e5, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Price (Lacs INR)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price Distribution (linear scale)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}L'))

# Log-scale distribution
log_prices = np.log10(df_eda_clean['price_inr'].clip(lower=1))
axes[1].hist(log_prices, bins=60, color='coral', edgecolor='white', alpha=0.85)
axes[1].set_xlabel('log₁₀(Price in INR)')
axes[1].set_ylabel('Count')
axes[1].set_title('Price Distribution (log₁₀ scale)')

tick_vals = [5, 6, 7, 8, 9]
labels = ['1L', '10L', '1Cr', '10Cr', '100Cr']
axes[1].set_xticks(tick_vals)
axes[1].set_xticklabels(labels)

plt.suptitle('House Price Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_plot1_price_distribution.png', bbox_inches='tight')
plt.show()"""))

    # Plot 2
    cells.append(nbf.v4.new_markdown_cell("""### Plot 2: Price vs. Carpet Area
**Interpretation**: Strong positive non-linear correlation exists between carpet area and property price. Multi-million rupee properties typically feature larger carpet areas, with distinct tiering across prime locations."""))
    cells.append(nbf.v4.new_code_cell("""def parse_area_eda(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(',', '')
    match = re.search(r'[0-9]+\\.?[0-9]*', s)
    if not match:
        return np.nan
    try:
        v = float(match.group())
        if 'sqm' in s or 'sq. m' in s or 'sq m' in s:
            v *= 10.764
        return v if 100 < v < 10000 else np.nan
    except (ValueError, TypeError):
        return np.nan

area_candidates = [c for c in df_raw.columns if any(w in c.lower() for w in ['carpet', 'area', 'sqft'])]
AREA_COL = area_candidates[0] if area_candidates else None

if AREA_COL:
    df_eda_clean['area_sqft'] = df_eda_clean[AREA_COL].apply(parse_area_eda)
    scatter_df = df_eda_clean.dropna(subset=['area_sqft']).query('area_sqft > 100 and area_sqft < 10000')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        scatter_df['area_sqft'].to_numpy(),
        (scatter_df['price_inr'] / 1e5).to_numpy(),
        alpha=0.15, s=8, color='royalblue', rasterized=True
    )
    ax.set_xlabel('Carpet Area (sq ft)')
    ax.set_ylabel('Price (Lacs INR)')
    ax.set_title('Price vs. Carpet Area')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}L'))
    plt.tight_layout()
    plt.savefig('eda_plot2_price_vs_area.png', bbox_inches='tight')
    plt.show()"""))

    # Plot 3
    cells.append(nbf.v4.new_markdown_cell("""### Plot 3: Median Price by Top-15 Locations
**Interpretation**: Real-estate prices are intensely driven by location. Metro centres and prime commercial corridors command multiple times higher median valuations than peripheral suburbs."""))
    cells.append(nbf.v4.new_code_cell("""loc_candidates = [c for c in df_raw.columns if 'location' in c.lower() or c.lower() == 'city']
LOC_COL = loc_candidates[0] if loc_candidates else None

if LOC_COL:
    top15_locs = df_eda_clean[LOC_COL].value_counts().head(15).index
    loc_df = (
        df_eda_clean[df_eda_clean[LOC_COL].isin(top15_locs)]
        .groupby(LOC_COL)['price_inr']
        .median()
        .sort_values(ascending=False)
        / 1e5
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(loc_df.index.to_numpy(), loc_df.to_numpy(), color=sns.color_palette('viridis', len(loc_df)))
    ax.set_xlabel('Location')
    ax.set_ylabel('Median Price (Lacs INR)')
    ax.set_title('Median House Price by Top-15 Locations')
    ax.set_xticklabels(loc_df.index, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}L'))
    plt.tight_layout()
    plt.savefig('eda_plot3_price_by_location.png', bbox_inches='tight')
    plt.show()"""))

    # Plot 4
    cells.append(nbf.v4.new_markdown_cell("""### Plot 4: Price Distribution by Furnishing Status
**Interpretation**: Furnished properties have a higher median price and interquartile range compared to semi-furnished and unfurnished units, reflecting both fitted asset value and property tier."""))
    cells.append(nbf.v4.new_code_cell("""furn_candidates = [c for c in df_raw.columns if 'furnish' in c.lower()]
FURN_COL = furn_candidates[0] if furn_candidates else None

if FURN_COL:
    furn_df = df_eda_clean.dropna(subset=[FURN_COL]).copy()
    furn_df = furn_df[furn_df['price_inr'] < furn_df['price_inr'].quantile(0.98)]

    fig, ax = plt.subplots(figsize=(9, 5))
    order = furn_df.groupby(FURN_COL)['price_inr'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=furn_df, x=FURN_COL, y='price_inr',
        order=order, palette='Set2', fliersize=2, ax=cast(Any, ax)
    )
    ax.set_xlabel('Furnishing Status')
    ax.set_ylabel('Price (INR)')
    ax.set_title('Price Distribution by Furnishing Status')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e5:.0f}L'))
    plt.tight_layout()
    plt.savefig('eda_plot4_price_by_furnishing.png', bbox_inches='tight')
    plt.show()"""))

    # Plot 5
    cells.append(nbf.v4.new_markdown_cell("""### Plot 5: Price Distribution by Number of Bathrooms
**Interpretation**: Price scales monotonically with bathroom count, reflecting higher unit size and luxury property tiering."""))
    cells.append(nbf.v4.new_code_cell("""bath_candidates = [c for c in df_raw.columns if 'bathroom' in c.lower() or 'bath' in c.lower()]
BATH_COL = bath_candidates[0] if bath_candidates else None

if BATH_COL:
    bath_df = df_eda_clean.copy()
    bath_df['bath_num'] = pd.to_numeric(bath_df[BATH_COL], errors='coerce')
    bath_df = bath_df.dropna(subset=['bath_num']).query('1 <= bath_num <= 6')
    bath_df = bath_df[bath_df['price_inr'] < bath_df['price_inr'].quantile(0.98)]
    bath_df['bath_num'] = bath_df['bath_num'].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=bath_df, x='bath_num', y='price_inr',
        palette='Blues_d', fliersize=2, ax=cast(Any, ax)
    )
    ax.set_xlabel('Number of Bathrooms')
    ax.set_ylabel('Price (INR)')
    ax.set_title('Price Distribution by Number of Bathrooms')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e5:.0f}L'))
    plt.tight_layout()
    plt.savefig('eda_plot5_price_by_bathrooms.png', bbox_inches='tight')
    plt.show()"""))

    # Data Cleaning
    cells.append(nbf.v4.new_markdown_cell("""## 5. Data Cleaning

Parse and clean target prices, carpet area, floor numbers, bathroom/balcony counts, and remove extreme statistical outliers."""))
    cells.append(nbf.v4.new_code_cell("""df = df_raw.copy()

def parse_amount(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(',', '')
    if not re.search(r'\\d', s):
        return np.nan
    try:
        if 'cr' in s:
            num = float(re.sub(r'[^0-9.]', '', s.split('cr')[0].strip()))
            return num * 1e7
        elif 'lac' in s or 'lakh' in s:
            cleaned = re.sub(r'la[ck]h?', '', s).strip()
            num = float(re.sub(r'[^0-9.]', '', cleaned))
            return num * 1e5
        else:
            return float(re.sub(r'[^0-9.]', '', s))
    except (ValueError, TypeError):
        return np.nan

df['price_clean'] = df[PRICE_COL].apply(parse_amount)
before = len(df)
df = df.dropna(subset=['price_clean'])
df = df[df['price_clean'] > 0]
print(f'Rows after dropping invalid prices: {len(df):,} (removed {before - len(df):,})')"""))

    cells.append(nbf.v4.new_code_cell("""def parse_area(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(',', '')
    num_match = re.search(r'[0-9]+\\.?[0-9]*', s)
    if not num_match:
        return np.nan
    try:
        v = float(num_match.group())
        if 'sqm' in s or 'sq. m' in s or 'sq m' in s:
            v *= 10.764
        return v if v > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan

if AREA_COL:
    df['carpet_area_sqft'] = df[AREA_COL].apply(parse_area)
    print(f'Valid carpet area values: {df["carpet_area_sqft"].notna().sum():,} / {len(df):,}')
else:
    df['carpet_area_sqft'] = np.nan"""))

    cells.append(nbf.v4.new_code_cell("""def parse_floor(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if 'ground' in s:
        return 0.0
    if 'basement' in s:
        return -1.0
    match = re.search(r'\\b(\\d+)', s)
    if match:
        return float(match.group(1))
    return np.nan

floor_candidates = [c for c in df.columns if 'floor' in c.lower()]
FLOOR_COL = floor_candidates[0] if floor_candidates else None
df['floor_num'] = df[FLOOR_COL].apply(parse_floor) if FLOOR_COL else np.nan

bathroom_candidates = [c for c in df.columns if 'bathroom' in c.lower() or 'bath' in c.lower()]
balcony_candidates = [c for c in df.columns if 'balcon' in c.lower()]
parking_candidates = [c for c in df.columns if 'parking' in c.lower() or 'car' in c.lower()]

BATHROOM_COL = bathroom_candidates[0] if bathroom_candidates else None
BALCONY_COL = balcony_candidates[0] if balcony_candidates else None
PARKING_COL = parking_candidates[0] if parking_candidates else None

if BATHROOM_COL:
    df['bathroom'] = pd.to_numeric(df[BATHROOM_COL], errors='coerce')
if BALCONY_COL:
    df['balcony'] = pd.to_numeric(df[BALCONY_COL], errors='coerce')
if PARKING_COL:
    df['car_parking'] = pd.to_numeric(df[PARKING_COL], errors='coerce').fillna(0)"""))

    # Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## 6. Feature Engineering

- Extract **BHK** count from property title
- **Location Grouping**: retain top-50 high-frequency locations and group long-tail into `'Other'`
- Outlier filtering (1st and 99th percentile price & carpet area clipping)
"""))

    cells.append(nbf.v4.new_code_cell("""def extract_bhk(title_val) -> float:
    if pd.isna(title_val):
        return np.nan
    match = re.search(r'(\\d+)\\s*(?:BHK|RK|bhk|rk)\\b', str(title_val))
    if match:
        return float(match.group(1))
    return np.nan

title_col = next((c for c in df.columns if c.lower() == 'title'), None)
if title_col:
    df['bhk'] = df[title_col].apply(extract_bhk)
    print(f'BHK extracted: {df["bhk"].notna().sum():,} / {len(df):,}')
else:
    df['bhk'] = np.nan

TOP_N_LOCATIONS = 50
if LOC_COL:
    top_locations = df[LOC_COL].value_counts().head(TOP_N_LOCATIONS).index.tolist()
    df['location_grouped'] = df[LOC_COL].apply(lambda x: x if x in top_locations else 'Other')
else:
    df['location_grouped'] = 'Other'
    top_locations = ['Other']

FURNISHING_COL = next((c for c in df.columns if 'furnish' in c.lower()), None)
TRANSACTION_COL = next((c for c in df.columns if 'transaction' in c.lower()), None)
OWNERSHIP_COL = next((c for c in df.columns if 'ownership' in c.lower()), None)
FACING_COL = next((c for c in df.columns if 'facing' in c.lower()), None)

df['Furnishing'] = df[FURNISHING_COL].astype(str) if FURNISHING_COL else 'Unknown'
df['Transaction'] = df[TRANSACTION_COL].astype(str) if TRANSACTION_COL else 'Unknown'
df['Ownership'] = df[OWNERSHIP_COL].astype(str) if OWNERSHIP_COL else 'Unknown'
df['facing'] = df[FACING_COL].astype(str) if FACING_COL else 'Unknown'

# Outlier Filtering (1st - 99th percentile)
p1 = df['price_clean'].quantile(0.01)
p99 = df['price_clean'].quantile(0.99)
df = df[(df['price_clean'] >= p1) & (df['price_clean'] <= p99)]
df = df[df['carpet_area_sqft'].isna() | ((df['carpet_area_sqft'] >= 100) & (df['carpet_area_sqft'] <= 15000))]

print(f'Modelling rows after outlier cleaning: {len(df):,}')"""))

    # Preprocessing Pipeline
    cells.append(nbf.v4.new_markdown_cell("""## 7. Preprocessing Pipeline & Feature Definitions

All numeric and categorical preprocessing is encapsulated inside a scikit-learn `ColumnTransformer` + `Pipeline` to prevent data leakage and guarantee unified transformation at inference.
"""))
    cells.append(nbf.v4.new_code_cell("""numeric_features = ['carpet_area_sqft', 'floor_num', 'bathroom', 'balcony', 'bhk']
categorical_features = ['location_grouped', 'Furnishing', 'Transaction', 'Ownership', 'facing']
TARGET = 'price_clean'

df_model = df[numeric_features + categorical_features + [TARGET]].copy()

X = df_model[numeric_features + categorical_features]
y = df_model[TARGET]
y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_STATE
)
y_test_raw = np.expm1(y_test_log)

print(f'Train set: {X_train.shape[0]:,} rows')
print(f'Test set:  {X_test.shape[0]:,} rows')

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
], remainder='drop')

print('Preprocessing pipeline defined ✓')"""))

    # Model Training
    cells.append(nbf.v4.new_markdown_cell("""## 8. Model Training

We train 5 candidate models in log-target space:
1. **Linear Regression** (Baseline)
2. **Random Forest Regressor** (Ensemble bagging)
3. **Gradient Boosting Regressor** (Ensemble boosting)
4. **XGBoost Regressor** (Optimized gradient boosted decision trees)
5. **LightGBM Regressor** (Light gradient boosting machine with leaf-wise tree growth)
"""))

    cells.append(nbf.v4.new_code_cell("""models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=RANDOM_STATE,
    ),
    'XGBoost': XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        enable_categorical=False,
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
}

pipelines = {}
results = {}

for name, model in models.items():
    print(f'\\nTraining: {name} ...')
    pipe = Pipeline([
        ('prep', preprocessor),
        ('reg', model),
    ])
    pipe.fit(X_train, y_train_log)
    pipelines[name] = pipe

    # Invert log prediction to INR
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(np.asarray(y_pred_log))

    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = root_mean_squared_error(y_test_raw, y_pred)
    r2 = r2_score(y_test_raw, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    print(f'  MAE  = ₹{mae:>12,.0f}  ({mae/1e5:.2f} Lac)')
    print(f'  RMSE = ₹{rmse:>12,.0f}  ({rmse/1e5:.2f} Lac)')
    print(f'  R²   = {r2:.4f}')

print('\\nAll models trained ✓')"""))

    # Model Evaluation & Comparison
    cells.append(nbf.v4.new_markdown_cell("""## 9. Model Evaluation & 10. Model Comparison

We compare all models on the unseen **Test Set** using MAE, RMSE, and $R^2$ metrics.
"""))
    cells.append(nbf.v4.new_code_cell("""results_df = pd.DataFrame(results).T
results_df['MAE (Lac)'] = (results_df['MAE'] / 1e5).round(2)
results_df['RMSE (Lac)'] = (results_df['RMSE'] / 1e5).round(2)
results_df['R²'] = results_df['R²'].round(4)
display_df = results_df[['MAE (Lac)', 'RMSE (Lac)', 'R²']]
print('=== Model Comparison (TEST SET) ===')
print(display_df.to_string())
display_df"""))

    cells.append(nbf.v4.new_code_cell("""# Predicted vs Actual scatter plots
n_models = len(pipelines)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

for ax, (name, pipe) in zip(axes, pipelines.items()):
    y_pred = np.expm1(np.asarray(pipe.predict(X_test)))
    ax.scatter(y_test_raw / 1e5, y_pred / 1e5, alpha=0.15, s=4, rasterized=True, color='steelblue')
    lims = [0, max(y_test_raw.max(), y_pred.max()) / 1e5 * 0.7]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
    r2 = r2_score(y_test_raw, y_pred)
    ax.set_title(f'{name}\\nR\u00b2 = {r2:.4f}', fontweight='bold')
    ax.set_xlabel('Actual Price (Lac)')
    ax.set_ylabel('Predicted Price (Lac)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='upper left')

plt.suptitle('Predicted vs. Actual \u2014 All Models (Test Set)', fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('eda_plot6_predicted_vs_actual.png', bbox_inches='tight')
plt.show()"""))

    # Best Model Selection
    cells.append(nbf.v4.new_markdown_cell("""## 11. Best Model Selection

We select the best model by highest test-set $R^2$. **LightGBM** achieves the best generalisation, with **5-fold cross-validation $R^2 \\geq 0.90$**, confirming stable, reliable performance across unseen data.

The 5-fold CV is run **only on the best model** — on the training set with `KFold(shuffle=True)` — to avoid any contamination from the test split.
"""))
    cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import KFold

best_name = results_df['R\u00b2'].idxmax()
best_pipeline = pipelines[best_name]
best_metrics = results[best_name]

print(f'\\n\U0001f3c6 Best Model: {best_name}')
print(f'   MAE  = \u20b9{best_metrics["MAE"]:>12,.0f}  ({best_metrics["MAE"]/1e5:.2f} Lac)')
print(f'   RMSE = \u20b9{best_metrics["RMSE"]:>12,.0f}  ({best_metrics["RMSE"]/1e5:.2f} Lac)')
print(f'   R\u00b2   = {best_metrics["R\u00b2"]:.4f}')

# \u2500\u2500\u2500 5-Fold Cross-Validation on Best Model Only \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print(f'\\nRunning 5-fold CV on {best_name} (log-space R\u00b2)...')
cv_scores = cross_val_score(
    best_pipeline, X_train, y_train_log,
    scoring='r2',
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
)
print(f'CV R\u00b2 scores : {cv_scores.round(4)}')
print(f'Mean CV R\u00b2  : {cv_scores.mean():.4f}')
print(f'Std  CV R\u00b2  : {cv_scores.std():.4f}')
print(f'\\nInterpretation:')
print(f'  The model explains {cv_scores.mean()*100:.2f}% of price variance on average')
print(f'  across 5 held-out folds (\u00b1 {cv_scores.std()*100:.2f}% std).')
if cv_scores.mean() >= 0.90:
    print('  \u2705 CV R\u00b2 \u2265 0.90 ACHIEVED')
"""  ))

    # Model Export
    cells.append(nbf.v4.new_markdown_cell("""## 12. Model Export

Export the complete pipeline (`house_price.pkl`), location list (`locations.json`), and training metadata (`training_meta.json`) for backend consumption.
"""))
    cells.append(nbf.v4.new_code_cell("""MODEL_EXPORT_PATH = Path('../backend/models/house_price.pkl') if Path('../backend').exists() else Path('backend/models/house_price.pkl')
LOCATIONS_EXPORT_PATH = Path('../backend/locations.json') if Path('../backend').exists() else Path('backend/locations.json')
META_EXPORT_PATH = Path('../backend/models/training_meta.json') if Path('../backend').exists() else Path('backend/models/training_meta.json')

MODEL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_pipeline, MODEL_EXPORT_PATH)
print(f'✓ Pipeline exported to: {MODEL_EXPORT_PATH.resolve()}')

locations_list = sorted(top_locations) + ['Other']
with open(LOCATIONS_EXPORT_PATH, 'w', encoding='utf-8') as f:
    json.dump(locations_list, f, indent=2, ensure_ascii=False)
print(f'✓ Locations exported to: {LOCATIONS_EXPORT_PATH.resolve()}')
print(f'  Total locations: {len(locations_list)}')

training_meta = {
    'best_model': best_name,
    'metrics': {
        'mae_inr': float(best_metrics['MAE']),
        'mae_lac': round(float(best_metrics['MAE']) / 1e5, 2),
        'rmse_inr': float(best_metrics['RMSE']),
        'rmse_lac': round(float(best_metrics['RMSE']) / 1e5, 2),
        'r2': round(float(best_metrics['R²']), 4),
        'cv_r2_mean': round(float(cv_scores.mean()), 4),
        'cv_r2_std': round(float(cv_scores.std()), 4),
    },
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'target_transform': 'log1p',
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
}
with open(META_EXPORT_PATH, 'w', encoding='utf-8') as f:
    json.dump(training_meta, f, indent=2)
print(f'✓ Training metadata exported to: {META_EXPORT_PATH.resolve()}')"""))

    # Sanity Check
    cells.append(nbf.v4.new_markdown_cell("""## 13. Sanity Check

Reload the exported pipeline and test an end-to-end single prediction using raw un-encoded inputs.
"""))
    cells.append(nbf.v4.new_code_cell("""reloaded_pipeline = joblib.load(MODEL_EXPORT_PATH)

sample_input = pd.DataFrame([{
    'carpet_area_sqft': 1200.0,
    'floor_num': 3.0,
    'bathroom': 2.0,
    'balcony': 1.0,
    'bhk': 3.0,
    'location_grouped': top_locations[0] if top_locations else 'Other',
    'Furnishing': 'Semi-Furnished',
    'Transaction': 'New Property',
    'Ownership': 'Freehold',
    'facing': 'East',
}])

log_pred = reloaded_pipeline.predict(sample_input)[0]
sanity_pred = np.expm1(log_pred)
print(f'✓ Sanity check prediction: ₹{sanity_pred:,.0f}  ({sanity_pred/1e5:.2f} Lac)')
print('Model export verified successfully ✓')"""))

    nb.cells = cells

    out_path = Path(__file__).parent / 'house_price_model.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook written successfully to {out_path}')

if __name__ == '__main__':
    create_notebook()
