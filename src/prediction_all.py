import numpy as np
from catboost import CatBoostRegressor
from itertools import product
import time

#############################################
# models uploading
#############################################
model_paths = [
    #'catboost_DDist90%_0.cbm',
    'catboost_DDist90%_1.cbm',
    'catboost_DDist90%_2.cbm',
    'catboost_DDist90%_3.cbm',
    'catboost_DDist90%_4.cbm',
    'catboost_DDist90%_5.cbm',
    'catboost_DDist90%_6.cbm',
    'catboost_DDist90%_7.cbm',
    'catboost_DDist90%_8.cbm'
]

models = []

for path in model_paths:
    m = CatBoostRegressor(thread_count=-1)
    m.load_model(path)
    models.append(m)


# ==========================================================
# Definition of ranges and number of sambels in every range
# ==========================================================
a, b, c = 25, 15, 3
FS = np.linspace(1, 5, b)
SV = np.linspace(10, 20, a)
SF = np.linspace(0.5, 3, b)
AT = np.linspace(10, 18, c)
HU = np.linspace(25, 40, c)
WS = np.linspace(0.5, 5, a)
WA = np.linspace(-80, 10, a)
WCV = np.linspace(-180, 150, a)

cols = ['FS', 'SV', 'SF', 'AT', 'HU', 'WS', 'WA', 'WCV']

# ==========================================================
# Batch prediction without storage
# ==========================================================
batch_limit = 30_000
combinations = product(FS, SV, SF, AT, HU, WS, WA, WCV)

batch = []
count = 0
min_pred = float('inf')
min_params = None

start = time.time()

# ==========================================================
# prediction using all models and taking the average
# ==========================================================
for combo in combinations:
    batch.append(combo)

    if len(batch) >= batch_limit:
        X = np.array(batch, dtype=np.float32)

        
        preds_sum = np.zeros(len(X), dtype=np.float32)
        for m in models:
            preds_sum += m.predict(X)

        preds_avg = preds_sum / len(models)

        
        local_min_idx = np.argmin(preds_avg)
        if preds_avg[local_min_idx] < min_pred:
            min_pred = preds_avg[local_min_idx]
            min_params = X[local_min_idx].copy()

        count += len(batch)
        print(f"processed {count:,} sambels... ( the minimum until now  {min_pred:.6f})")
        batch = []


if batch:
    X = np.array(batch, dtype=np.float32)
    preds_sum = np.zeros(len(X), dtype=np.float32)
    for m in models:
        preds_sum += m.predict(X)
    preds_avg = preds_sum / len(models)
    local_min_idx = np.argmin(preds_avg)
    if preds_avg[local_min_idx] < min_pred:
        min_pred = preds_avg[local_min_idx]
        min_params = X[local_min_idx].copy()
    count += len(batch)

end = time.time()

# ==========================================================
#show resultes
# ==========================================================
print("\n  processing done ")
print(f" time of processing : {end - start:.2f} seconds")
print(f"   the number of all sambels : {count:,}")
print(f"\n minimum value : {min_pred:.6f}\n")

for name, val in zip(cols, min_params):
    print(f"{name}: {val:.3f}")