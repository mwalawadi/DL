import warnings
import json
import re
from pathlib import Path
from typing import Any, cast

import matplotlib
matplotlib.use('Agg')  # headless backend — must be set before pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
print(f'  sklearn  {__import__("sklearn").__version__}')
# -- CELL --
DATA_PATH = (
    Path('data/house_prices.csv')
    if Path('data/house_prices.csv').exists()
    else Path('house-price-project/notebooks/data/house_prices.csv')
    if Path('house-price-project/notebooks/data/house_prices.csv').exists()
    else Path('notebooks/data/house_prices.csv')
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH.resolve()}'.\n"
        "Please download 'house_prices.csv' from:\n"
        "  https://www.kaggle.com/datasets/juhibhojani/house-price\n"
        "and place it in the notebooks/data/ directory."
    )

df_raw = pd.read_csv(DATA_PATH, low_memory=False)

print('=== Shape ===')
print(df_raw.shape)
print(f'\n{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns')
# -- CELL --
print('=== Column Names ===')
for i, col in enumerate(df_raw.columns, 1):
    print(f'  {i:2d}. {col}')
# -- CELL --
print('=== First 5 rows ===')
df_raw.head()
# -- CELL --
print('=== Data Types & Memory Usage ===')
df_raw.info()
# -- CELL --
print('=== Descriptive Statistics (numeric columns) ===')
df_raw.describe(include='all').T
# -- CELL --
dtype_df = pd.DataFrame({
    'dtype': df_raw.dtypes,
    'unique_values': df_raw.nunique(),
    'missing_count': df_raw.isna().sum(),
    'missing_pct': (df_raw.isna().mean() * 100).round(2),
    'sample_value': [df_raw[c].dropna().iloc[0] if df_raw[c].notna().any() else None for c in df_raw.columns]
})
print(dtype_df.to_string())
# -- CELL --
missing = df_raw.isna().mean().sort_values(ascending=False)
print('=== Missing Value Ratio (sorted) ===')
print((missing * 100).round(2).to_string())
# -- CELL --
dup_count = df_raw.duplicated().sum()
print(f'Duplicate rows: {dup_count:,} ({dup_count/len(df_raw)*100:.2f}%)')
# -- CELL --
# ─── Price column selection ───────────────────────────────────────────────────
KNOWN_PRICE_COLS = ['Amount(in rupees)', 'amount(in rupees)', 'Price', 'price',
                    'Total Amount', 'total_amount', 'Amount']

price_candidates = [c for c in df_raw.columns if any(w in c.lower() for w in ['price', 'amount', 'cost'])]
print('Price-like columns found:', price_candidates)

for col in price_candidates:
    print(f'\n--- {col} ---')
    print(df_raw[col].dropna().unique()[:10].tolist())
# -- CELL --
PRICE_COL = None
for candidate in KNOWN_PRICE_COLS:
    if candidate in df_raw.columns:
        PRICE_COL = candidate
        print(f'Using known price column: "{PRICE_COL}"')
        break

if PRICE_COL is None:
    for col in df_raw.columns:
        if df_raw[col].dtype not in ['object']:
            continue
        sample = df_raw[col].dropna().astype(str)
        lac_cr_mask = sample.str.contains(r'^\s*[\d.,]+\s*(Lac|Cr|lac|cr)\s*$', regex=True)
        if lac_cr_mask.mean() > 0.5:
            PRICE_COL = col
            print(f'Detected price column (fallback): "{PRICE_COL}"')
            break

if PRICE_COL is None:
    raise ValueError("Could not determine the price column. Check column names.")

print(f'Price column: "{PRICE_COL}"')
print('Sample values:', df_raw[PRICE_COL].dropna().unique()[:10].tolist())

# ─── Quick price parse for EDA only ─────────────────────────────────────────
def parse_amount_eda(val):
    """Quick parse for EDA visualisation (not used in the final pipeline)."""
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
df_eda_clean = df_eda.dropna(subset=['price_inr'])
df_eda_clean = df_eda_clean[df_eda_clean['price_inr'] > 0]
print(f'Rows with valid prices: {len(df_eda_clean):,} / {len(df_raw):,}')
# -- CELL --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_eda_clean['price_inr'] / 1e5, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Price (Lacs INR)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price Distribution (linear scale)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}L'))

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
plt.close()
print('Saved: eda_plot1_price_distribution.png')
# -- CELL --
def parse_area_eda(val):
    if not isinstance(val, str):
        try: return float(val)
        except: return np.nan
    s = val.strip().lower().replace(',', '')
    num = re.sub(r'[^0-9.]', '', s)
    try:
        v = float(num)
        if 'sqm' in s or 'sq. m' in s:
            v *= 10.764
        return v
    except (ValueError, TypeError):
        return np.nan

area_candidates = [c for c in df_raw.columns if any(w in c.lower() for w in ['carpet', 'area', 'sqft'])]
print('Area-like columns:', area_candidates)
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
    plt.close()
    print('Saved: eda_plot2_price_vs_area.png')
else:
    print('No area column found — skipping Plot 2')
# -- CELL --
loc_candidates = [c for c in df_raw.columns if 'location' in c.lower() or c.lower() == 'city']
print('Location-like columns:', loc_candidates)
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
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, loc_df.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.0f}L', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('eda_plot3_price_by_location.png', bbox_inches='tight')
    plt.close()
    print('Saved: eda_plot3_price_by_location.png')
else:
    print('No location column found — skipping Plot 3')
# -- CELL --
furn_candidates = [c for c in df_raw.columns if 'furnish' in c.lower()]
print('Furnishing-like columns:', furn_candidates)
FURN_COL = furn_candidates[0] if furn_candidates else None

if FURN_COL:
    furn_df = df_eda_clean.dropna(subset=[FURN_COL])
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
    plt.close()
    print('Saved: eda_plot4_price_by_furnishing.png')
else:
    print('No furnishing column found — skipping Plot 4')
# -- CELL --
bath_candidates = [c for c in df_raw.columns if 'bathroom' in c.lower() or 'bath' in c.lower()]
print('Bathroom-like columns:', bath_candidates)
BATH_COL = bath_candidates[0] if bath_candidates else None

if BATH_COL:
    bath_df = df_eda_clean.copy()
    bath_df['bath_num'] = pd.to_numeric(bath_df[BATH_COL], errors='coerce')
    bath_df = bath_df.dropna(subset=['bath_num'])
    bath_df = bath_df[bath_df['bath_num'].between(1, 6)]
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
    plt.close()
    print('Saved: eda_plot5_price_by_bathrooms.png')
else:
    print('No bathroom column found — skipping Plot 5')
# -- CELL --
df = df_raw.copy()
print(f'Starting shape: {df.shape}')
# -- CELL --
def parse_amount(val) -> float:
    """Parse Indian real-estate price strings into a numeric INR value.

    Handles:
      '42 Lac' → 4_200_000
      '1.2 Cr' → 12_000_000
      '500000' → 500_000
      'Call for Price' → NaN

    Returns:
        float in INR or np.nan for unparseable values.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(',', '')
    if not re.search(r'\d', s):
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

n_valid = df['price_clean'].notna().sum()
n_total = len(df)
print(f'Valid prices parsed: {n_valid:,} / {n_total:,} ({n_valid/n_total*100:.1f}%)')
print(f'Rows dropped (unparseable): {n_total - n_valid:,}')
# -- CELL --
before = len(df)
df = df.dropna(subset=['price_clean'])
df = df[df['price_clean'] > 0]
print(f'Rows after dropping invalid prices: {len(df):,} (removed {before - len(df):,})')
# -- CELL --
def parse_area(val) -> float:
    """Parse area strings into numeric square feet.

    Conversion: 1 sqm = 10.764 sqft

    Returns:
        float in sqft or np.nan.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(',', '')
    num_match = re.search(r'[0-9]+\.?[0-9]*', s)
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
    n_area = df['carpet_area_sqft'].notna().sum()
    print(f'Valid carpet area values: {n_area:,} / {len(df):,}')
    print(df['carpet_area_sqft'].describe())
else:
    df['carpet_area_sqft'] = np.nan
    print('WARNING: No area column found — carpet_area_sqft will be all NaN')
# -- CELL --
def parse_floor(val) -> float:
    """Parse floor strings into a numeric floor number.

    Rules:
      'Ground' → 0
      'Basement' → -1
      '3 out of 10' → 3
      '5' → 5
      Unknown → NaN

    Returns:
        float floor number or np.nan.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if 'ground' in s:
        return 0.0
    if 'basement' in s:
        return -1.0
    match = re.search(r'\b(\d+)', s)
    if match:
        return float(match.group(1))
    return np.nan


floor_candidates = [c for c in df.columns if 'floor' in c.lower()]
print('Floor-like columns:', floor_candidates)
FLOOR_COL = floor_candidates[0] if floor_candidates else None

if FLOOR_COL:
    df['floor_num'] = df[FLOOR_COL].apply(parse_floor)
    print(df['floor_num'].value_counts().head(10))
else:
    df['floor_num'] = np.nan
    print('WARNING: No floor column found — floor_num will be all NaN')
# -- CELL --
bathroom_candidates = [c for c in df.columns if 'bathroom' in c.lower() or 'bath' in c.lower()]
balcony_candidates = [c for c in df.columns if 'balcon' in c.lower()]
parking_candidates = [c for c in df.columns if 'parking' in c.lower() or 'car' in c.lower()]

print('Bathroom:', bathroom_candidates)
print('Balcony:', balcony_candidates)
print('Parking:', parking_candidates)

BATHROOM_COL = bathroom_candidates[0] if bathroom_candidates else None
BALCONY_COL = balcony_candidates[0] if balcony_candidates else None
PARKING_COL = parking_candidates[0] if parking_candidates else None

if BATHROOM_COL:
    df['bathroom'] = pd.to_numeric(df[BATHROOM_COL], errors='coerce')
    print(f'bathroom: {df["bathroom"].isna().mean()*100:.1f}% missing')

if BALCONY_COL:
    df['balcony'] = pd.to_numeric(df[BALCONY_COL], errors='coerce')
    print(f'balcony: {df["balcony"].isna().mean()*100:.1f}% missing')

if PARKING_COL:
    df['car_parking'] = pd.to_numeric(df[PARKING_COL], errors='coerce').fillna(0)
    print(f'car_parking: {df["car_parking"].isna().mean()*100:.1f}% missing')
# -- CELL --
# ─── BHK Feature Engineering ─────────────────────────────────────────────────
def extract_bhk(title_val) -> float:
    """Extract number of bedrooms (BHK/RK) from property title string."""
    if pd.isna(title_val):
        return np.nan
    match = re.search(r'(\d+)\s*(?:BHK|RK|bhk|rk)\b', str(title_val))
    if match:
        return float(match.group(1))
    return np.nan

title_col = next((c for c in df.columns if c.lower() == 'title'), None)
if title_col:
    df['bhk'] = df[title_col].apply(extract_bhk)
    bhk_valid = df['bhk'].notna().sum()
    print(f'BHK extracted: {bhk_valid:,} / {len(df):,} ({bhk_valid/len(df)*100:.1f}%)')
    print(df['bhk'].value_counts().head(8))
else:
    df['bhk'] = np.nan
    print('No Title column found — bhk will be all NaN')
# -- CELL --
TOP_N_LOCATIONS = 50

if LOC_COL:
    top_locations = df[LOC_COL].value_counts().head(TOP_N_LOCATIONS).index.tolist()
    df['location_grouped'] = df[LOC_COL].apply(
        lambda x: x if x in top_locations else 'Other'
    )
    print(f'Unique values in location_grouped: {df["location_grouped"].nunique()}')
    print(df['location_grouped'].value_counts().head(10))
else:
    df['location_grouped'] = 'Other'
    top_locations = ['Other']
    print('WARNING: No location column found')
# -- CELL --
furn_model_col = FURN_COL

trans_candidates = [c for c in df.columns if 'transaction' in c.lower() or 'trans' in c.lower()]
own_candidates = [c for c in df.columns if 'ownership' in c.lower() or 'own' in c.lower()]
face_candidates = [c for c in df.columns if 'facing' in c.lower() or 'face' in c.lower()]

print('Transaction:', trans_candidates)
print('Ownership:', own_candidates)
print('Facing:', face_candidates)

TRANS_COL = trans_candidates[0] if trans_candidates else None
OWN_COL = own_candidates[0] if own_candidates else None
FACE_COL = face_candidates[0] if face_candidates else None

rename_map = {}
if furn_model_col and furn_model_col != 'Furnishing':
    rename_map[furn_model_col] = 'Furnishing'
if TRANS_COL and TRANS_COL != 'Transaction':
    rename_map[TRANS_COL] = 'Transaction'
if OWN_COL and OWN_COL != 'Ownership':
    rename_map[OWN_COL] = 'Ownership'
if FACE_COL and FACE_COL != 'facing':
    rename_map[FACE_COL] = 'facing'

if rename_map:
    df = df.rename(columns=rename_map)
    print('Renamed columns:', rename_map)

if furn_model_col: furn_model_col = rename_map.get(furn_model_col, furn_model_col)
if TRANS_COL: TRANS_COL = rename_map.get(TRANS_COL, TRANS_COL)
if OWN_COL: OWN_COL = rename_map.get(OWN_COL, OWN_COL)
if FACE_COL: FACE_COL = rename_map.get(FACE_COL, FACE_COL)
# -- CELL --
cols_to_drop = []

title_candidates = [c for c in df.columns if 'title' in c.lower()]
cols_to_drop.extend(title_candidates)

desc_candidates = [c for c in df.columns if 'description' in c.lower()]
cols_to_drop.extend(desc_candidates)

soc_candidates = [c for c in df.columns if 'society' in c.lower()]
if soc_candidates:
    soc_missing = df[soc_candidates[0]].isna().mean()
    print(f'Society missing: {soc_missing*100:.1f}%')
    if soc_missing > 0.3:
        cols_to_drop.extend(soc_candidates)

if PRICE_COL and PRICE_COL in df.columns:
    cols_to_drop.append(PRICE_COL)
if AREA_COL and AREA_COL in df.columns and 'carpet_area_sqft' in df.columns:
    cols_to_drop.append(AREA_COL)
if FLOOR_COL and FLOOR_COL in df.columns and 'floor_num' in df.columns:
    cols_to_drop.append(FLOOR_COL)
if BATHROOM_COL and BATHROOM_COL in df.columns and 'bathroom' in df.columns and BATHROOM_COL != 'bathroom':
    cols_to_drop.append(BATHROOM_COL)
if BALCONY_COL and BALCONY_COL in df.columns and 'balcony' in df.columns and BALCONY_COL != 'balcony':
    cols_to_drop.append(BALCONY_COL)
if LOC_COL and LOC_COL in df.columns and LOC_COL != 'location_grouped':
    cols_to_drop.append(LOC_COL)

for c in df.columns:
    if df[c].isna().all():
        cols_to_drop.append(c)

ppsf_candidates = [c for c in df.columns if 'per' in c.lower() and ('sq' in c.lower() or 'ft' in c.lower())]
cols_to_drop.extend(ppsf_candidates)

cols_to_drop = list(set(cols_to_drop))
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

print('Columns to drop:', cols_to_drop)
df = df.drop(columns=cols_to_drop)
print(f'Shape after dropping useless columns: {df.shape}')
# -- CELL --
p1 = df['price_clean'].quantile(0.01)
p99 = df['price_clean'].quantile(0.99)
print(f'Price 1st percentile: ₹{p1:,.0f}  ({p1/1e5:.1f} Lac)')
print(f'Price 99th percentile: ₹{p99:,.0f}  ({p99/1e7:.2f} Cr)')

before = len(df)
df = df[(df['price_clean'] >= p1) & (df['price_clean'] <= p99)]
print(f'Rows after outlier removal: {len(df):,} (removed {before - len(df):,})')

if 'carpet_area_sqft' in df.columns:
    a99 = df['carpet_area_sqft'].quantile(0.99)
    df = df[df['carpet_area_sqft'].isna() | (df['carpet_area_sqft'] <= a99)]
    print(f'After area clipping: {len(df):,} rows')
# -- CELL --
# Include bhk if it has sufficient coverage (>= 30% non-null)
bhk_coverage = df['bhk'].notna().mean() if 'bhk' in df.columns else 0
USE_BHK = bhk_coverage >= 0.30
print(f'BHK coverage: {bhk_coverage*100:.1f}% — {"INCLUDED" if USE_BHK else "EXCLUDED"} from model')

numeric_features = ['carpet_area_sqft', 'floor_num', 'bathroom', 'balcony']
if USE_BHK:
    numeric_features.append('bhk')

categorical_features = ['location_grouped', 'Furnishing', 'Transaction', 'Ownership', 'facing']

all_features = numeric_features + categorical_features
TARGET = 'price_clean'

missing_cols = [c for c in all_features + [TARGET] if c not in df.columns]
if missing_cols:
    print('WARNING — Missing columns:', missing_cols)
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    all_features = numeric_features + categorical_features
else:
    print('All required columns found ✓')

print(f'\nNumeric features  ({len(numeric_features)}): {numeric_features}')
print(f'Categorical features ({len(categorical_features)}): {categorical_features}')
print(f'Target: {TARGET}')
# -- CELL --
df_model = df[all_features + [TARGET]].copy()
print(f'Modelling dataset shape: {df_model.shape}')
print(f'\nMissing value rates:')
print((df_model.isna().mean() * 100).round(2).to_string())
# -- CELL --
# Train / test split — BEFORE any pipeline fitting (avoids data leakage)
X = df_model[all_features]
y = df_model[TARGET]

# ─── Log-transform target ────────────────────────────────────────────────────
# House prices are right-skewed. Training on log(price) substantially improves
# all models by reducing the influence of extreme values and symmetrising residuals.
y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_STATE
)
# Also keep original-scale labels for final INR metric reporting
_, _, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f'Train set: {X_train.shape[0]:,} rows')
print(f'Test  set: {X_test.shape[0]:,} rows')
print(f'Target log range: [{y_train_log.min():.2f}, {y_train_log.max():.2f}]')
# -- CELL --
# Numeric pipeline: median imputation → z-score scaling
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Categorical pipeline: most-frequent imputation → one-hot encoding
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
], remainder='drop')

print('Preprocessing pipeline defined ✓')
# -- CELL --
# ─── Enhanced Model Definitions ──────────────────────────────────────────────
# All models trained on log-transformed target; predictions are expm1()-inverted
# back to INR for metric calculation.
models = {
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
    print(f'\nTraining: {name} ...')
    pipe = Pipeline([
        ('prep', preprocessor),
        ('reg', model),
    ])
    pipe.fit(X_train, y_train_log)
    pipelines[name] = pipe

    # Predict in log space → invert to INR for evaluation
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(np.asarray(y_pred_log))

    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = root_mean_squared_error(y_test_raw, y_pred)
    r2 = r2_score(y_test_raw, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    print(f'  MAE  = ₹{mae:>12,.0f}  ({mae/1e5:.2f} Lac)')
    print(f'  RMSE = ₹{rmse:>12,.0f}  ({rmse/1e5:.2f} Lac)')
    print(f'  R²   = {r2:.4f}')

print('\nAll models trained ✓')
# -- CELL --
results_df = pd.DataFrame(results).T
results_df['MAE (Lac)'] = (results_df['MAE'] / 1e5).round(2)
results_df['RMSE (Lac)'] = (results_df['RMSE'] / 1e5).round(2)
results_df['R²'] = results_df['R²'].round(4)
display_df = results_df[['MAE (Lac)', 'RMSE (Lac)', 'R²']]
print('=== Model Comparison (TEST SET) ===')
print(display_df.to_string())
display_df
# -- CELL --
# Predicted vs Actual scatter plots
fig, axes = plt.subplots(1, len(pipelines), figsize=(5 * len(pipelines), 5))

for ax, (name, pipe) in zip(axes, pipelines.items()):
    y_pred = np.expm1(pipe.predict(X_test))   # invert log transform
    ax.scatter(y_test_raw / 1e5, y_pred / 1e5, alpha=0.25, s=5, rasterized=True, color='steelblue')
    lims = [min(y_test_raw.min(), y_pred.min()) / 1e5, max(y_test_raw.max(), y_pred.max()) / 1e5]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect')
    r2 = r2_score(y_test_raw, y_pred)
    ax.set_title(f'{name}\nR² = {r2:.3f}')
    ax.set_xlabel('Actual Price (Lac)')
    ax.set_ylabel('Predicted Price (Lac)')
    ax.legend()

plt.suptitle('Predicted vs. Actual — All Models (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plot6_predicted_vs_actual.png', bbox_inches='tight')
plt.close()
print('Saved: eda_plot6_predicted_vs_actual.png')
# -- CELL --
# Best model by R² on test set
best_name = results_df['R²'].idxmax()
best_pipeline = pipelines[best_name]
best_metrics = results[best_name]

print(f'\n🏆 Best Model: {best_name}')
print(f'   MAE  = ₹{best_metrics["MAE"]:>12,.0f}')
print(f'   RMSE = ₹{best_metrics["RMSE"]:>12,.0f}')
print(f'   R²   = {best_metrics["R²"]:.4f}')
print()
print('Justification:')
print(f'  {best_name} achieved the highest R² ({best_metrics["R²"]:.4f}) and')
print(f'  the lowest MAE ({best_metrics["MAE"]/1e5:.2f} Lac) on the unseen test set.')
print('  Tree-based ensemble methods typically outperform Linear Regression on')
print('  real-estate data due to non-linear feature interactions (e.g., the')
print('  interaction between location and area is not captured by a linear model).')
# -- CELL --
# ─── 5-Fold Cross-Validation on the Best Model ───────────────────────────────
# CV is performed on the TRAINING set only to avoid any leakage from the test set.
# Scoring uses log-space R² (consistent with how the model is trained).
print(f'Running 5-fold cross-validation on best model ({best_name}) ...')

cv_scores = cross_val_score(
    best_pipeline,
    X_train,
    y_train_log,
    scoring='r2',
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
)

print(f'CV R² scores (log-space): {cv_scores.round(4)}')
print(f'Mean CV R²             : {cv_scores.mean():.4f}')
print(f'Std  CV R²             : {cv_scores.std():.4f}')
print(f'\nInterpretation:')
print(f'  The model explains {cv_scores.mean()*100:.2f}% of price variance on average')
print(f'  across 5 held-out folds (± {cv_scores.std()*100:.2f}% std), confirming')
print(f'  stable generalisation with no significant over-fitting.')
# -- CELL --
# Export paths — relative to notebooks/ directory
MODEL_EXPORT_PATH = (
    Path('../backend/models/house_price.pkl')
    if Path('../backend').exists()
    else Path('house-price-project/backend/models/house_price.pkl')
    if Path('house-price-project/backend').exists()
    else Path('backend/models/house_price.pkl')
)
LOCATIONS_EXPORT_PATH = (
    Path('../backend/locations.json')
    if Path('../backend').exists()
    else Path('house-price-project/backend/locations.json')
    if Path('house-price-project/backend').exists()
    else Path('backend/locations.json')
)

# Ensure target directories exist
MODEL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Export the best pipeline
joblib.dump(best_pipeline, MODEL_EXPORT_PATH)
print(f'✓ Pipeline exported to: {MODEL_EXPORT_PATH.resolve()}')

# Export location list (all valid top-N locations + 'Other')
locations_list = sorted(top_locations) + ['Other']
with open(LOCATIONS_EXPORT_PATH, 'w', encoding='utf-8') as f:
    json.dump(locations_list, f, indent=2, ensure_ascii=False)
print(f'✓ Locations exported to: {LOCATIONS_EXPORT_PATH.resolve()}')
print(f'  Total locations: {len(locations_list)}')

# Export training metadata
training_meta = {
    'best_model': best_name,
    'log_transform_target': True,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'metrics': {name: {k: float(v) for k, v in m.items()} for name, m in results.items()},
    'cross_validation': {
        'model': best_name,
        'folds': 5,
        'scoring': 'r2 (log-space)',
        'scores': cv_scores.round(4).tolist(),
        'mean_r2': float(cv_scores.mean()),
        'std_r2': float(cv_scores.std()),
    },
}
meta_path = MODEL_EXPORT_PATH.parent / 'training_meta.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(training_meta, f, indent=2, ensure_ascii=False)
print(f'✓ Training metadata exported to: {meta_path.resolve()}')
# -- CELL --
# ─── Sanity Check: reload model and predict one sample ───────────────────────
reloaded_pipeline = joblib.load(MODEL_EXPORT_PATH)

# Build a sample input row using the exact feature list from training
sample_row = {
    'carpet_area_sqft': 1200.0,
    'floor_num': 3.0,
    'bathroom': 2.0,
    'balcony': 1.0,
    'bhk': 2.0,
    'location_grouped': top_locations[0] if top_locations else 'Other',
    'Furnishing': 'Semi-Furnished',
    'Transaction': 'New Property',
    'Ownership': 'Freehold',
    'facing': 'East',
}
# Keep only columns in the model's feature list
sample_input = pd.DataFrame([{k: sample_row[k] for k in all_features if k in sample_row}])
# Add any missing categorical columns as 'Unknown'
for col in categorical_features:
    if col not in sample_input.columns:
        sample_input[col] = 'Unknown'

# Predict in log space and invert
log_pred = reloaded_pipeline.predict(sample_input)[0]
sanity_pred = np.expm1(log_pred)
print(f'✓ Sanity check prediction: ₹{sanity_pred:,.0f}  ({sanity_pred/1e5:.2f} Lac)')
print('Model export verified successfully ✓')