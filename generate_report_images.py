import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (6, 3) # Much smaller size
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 8

# Create output directory
output_dir = 'report_images'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv('data_weekly_260120.csv')
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)

# Data verification info
data_source = "Source: data_weekly_260120.csv"
date_range = f"{df.index.min().date()} ~ {df.index.max().date()}"
footer_text = f"{data_source} | Period: {date_range}"

# 1. Nickel Price Time Series (Section 3.1)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['Com_LME_Ni_Cash'], color='navy', linewidth=1.2)
plt.title('LME Nickel Cash Price (2013-2026)', fontsize=12, fontweight='bold')
plt.xlabel('')
plt.ylabel('Price (USD/Ton)')
plt.grid(True, alpha=0.3)
plt.figtext(0.99, 0.01, footer_text, ha='right', fontsize=8, color='gray')

# Add annotations for key events mentioned in the report
events = [
    ('2016-01-01', 8000, 'Supply Glut'),
    ('2022-03-01', 45000, 'Russia-Ukraine'),
    ('2020-03-01', 11000, 'COVID-19')
]

for date_str, price, label in events:
    try:
        date_ts = pd.Timestamp(date_str)
        # Find closest date in index
        closest_date = df.index[df.index.get_indexer([date_ts], method='nearest')[0]]
        if closest_date in df.index:
            closest_price = df.loc[closest_date, 'Com_LME_Ni_Cash']
            
            plt.annotate(label, 
                         xy=(closest_date, closest_price), 
                         xytext=(closest_date, closest_price + (8000 if price > 20000 else 5000)),
                         arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.7),
                         horizontalalignment='center', fontsize=9)
    except:
        pass

plt.tight_layout()
plt.savefig(f'{output_dir}/nickel_price_ts.png', dpi=150) # Reduced DPI
plt.close()
print(f"Generated {output_dir}/nickel_price_ts.png")

# 2. Returns Distribution (Section 3.2)
# Calculate log returns
df['log_ret'] = np.log(df['Com_LME_Ni_Cash'] / df['Com_LME_Ni_Cash'].shift(1))

plt.figure(figsize=(8, 4))
plt.hist(df['log_ret'].dropna(), bins=60, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.title('Weekly Log Returns Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.figtext(0.99, 0.01, footer_text, ha='right', fontsize=8, color='gray')

# Add stats
mu = df['log_ret'].mean()
sigma = df['log_ret'].std()
plt.axvline(mu, color='red', linestyle='--', linewidth=1, label=f'Mean: {mu:.4f}')
plt.axvline(mu - 1.96*sigma, color='orange', linestyle=':', linewidth=1, label='95% CI')
plt.axvline(mu + 1.96*sigma, color='orange', linestyle=':', linewidth=1)
plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/returns_dist.png', dpi=150)
plt.close()
print(f"Generated {output_dir}/returns_dist.png")

# 3. Volatility (Section 3.3)
# 12-week rolling std dev, annualized (sqrt(52))
volatility = df['log_ret'].rolling(window=12).std() * np.sqrt(52)

plt.figure(figsize=(8, 4))
plt.plot(df.index, volatility, color='darkred', linewidth=1.2)
plt.title('Annualized Volatility (12-Week Rolling)', fontsize=12, fontweight='bold')
plt.ylabel('Volatility (Annualized)')
plt.xlabel('')
plt.grid(True, alpha=0.3)
plt.figtext(0.99, 0.01, footer_text, ha='right', fontsize=8, color='gray')

# Filter for plotting range to match report description generally
plt.fill_between(df.index, volatility, 0, alpha=0.1, color='red')

plt.tight_layout()
plt.savefig(f'{output_dir}/volatility_ts.png', dpi=150)
plt.close()
print(f"Generated {output_dir}/volatility_ts.png")
