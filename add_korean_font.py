#!/usr/bin/env python3
"""
Script to add Korean font configuration to sparta2_advanced.ipynb
"""
import json

NOTEBOOK_PATH = "/Users/jaeholee/Desktop/sparta_2/sparta2_advanced.ipynb"

def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the first import cell and add Korean font config
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'import matplotlib.pyplot as plt' in source and 'import pandas' in source:
                # This is the main imports cell - add Korean font configuration
                new_source = [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import matplotlib.font_manager as fm\n",
                    "import seaborn as sns\n",
                    "import xgboost as xgb\n",
                    "import lightgbm as lgb\n",
                    "from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor\n",
                    "from sklearn.linear_model import Ridge, LinearRegression\n",
                    "from sklearn.metrics import mean_squared_error\n",
                    "from sklearn.model_selection import TimeSeriesSplit\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# 한글 폰트 설정 (Mac)\n",
                    "plt.rcParams['font.family'] = 'AppleGothic'\n",
                    "plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지\n",
                    "\n",
                    "# 시각화 설정\n",
                    "plt.rcParams['figure.figsize'] = (12, 5)\n",
                    "plt.rcParams['font.size'] = 11\n",
                    "\n",
                    "# 상수 정의\n",
                    "SPARTA2_RMSE = 406.80\n",
                    "RANDOM_STATE = 42\n",
                    "np.random.seed(RANDOM_STATE)\n",
                ]
                
                # Preserve the rest of the cell (date settings, etc.)
                remaining_lines = []
                found_date = False
                for line in cell.get('source', []):
                    if '날짜 설정' in line or 'VAL_START' in line:
                        found_date = True
                    if found_date:
                        remaining_lines.append(line)
                
                new_source.extend(["\n"] + remaining_lines)
                notebook['cells'][i]['source'] = new_source
                notebook['cells'][i]['outputs'] = []  # Clear outputs
                print(f"Updated imports cell at index {i} with Korean font configuration")
                break
    
    # Write modified notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"Saved: {NOTEBOOK_PATH}")
    print("Korean font (AppleGothic) configuration added!")
    return True

if __name__ == "__main__":
    main()
