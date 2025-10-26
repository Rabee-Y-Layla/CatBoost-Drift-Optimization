import numpy as np
from catboost import CatBoostRegressor
from itertools import product
import time

##################################################
# upload CatBoost model
# ################################################
model = CatBoostRegressor(thread_count=-1)
model.load_model('catboost_DDist90%_6.cbm')

############################################################
# Definition of ranges and number of sambels in every range
############################################################
a=25
b=15
c=3  
FS = np.linspace(1, 5, b)
SV = np.linspace(10, 20, a)
SF = np.linspace(0.5, 3, b)
AT = np.linspace(10, 18, c)
HU = np.linspace(25, 40, c)
WS = np.linspace(0.5, 5, a)
WA = np.linspace(-80, 10, a)
WCV = np.linspace(0, 150, a)

cols = ['FS', 'SV', 'SF', 'AT', 'HU', 'WS', 'WA', 'WCV']

# ==========================================================
# Batch prediction without storage
# ==========================================================
batch_limit = 30000  # Batch size, you can increase it if your RAM is large
combinations = product(FS, SV, SF, AT, HU, WS, WA, WCV)

batch = []
count = 0
min_pred = float('inf')
min_params = None

start = time.time()

for combo in combinations:
    batch.append(combo)

    if len(batch) >= batch_limit:
        X = np.array(batch, dtype=np.float32)
        preds = model.predict(X)

        # find hte minimum of this Batch
        local_min_idx = np.argmin(preds)
        if preds[local_min_idx] < min_pred:
            min_pred = preds[local_min_idx]
            min_params = X[local_min_idx].copy()

        count += len(batch)
        print(f"processed {count:,} sambels... ( the minimum until now  {min_pred:.6f})")
        batch = []

#the last batch
if batch:
    X = np.array(batch, dtype=np.float32)
    preds = model.predict(X)
    local_min_idx = np.argmin(preds)
    if preds[local_min_idx] < min_pred:
        min_pred = preds[local_min_idx]
        min_params = X[local_min_idx].copy()
    count += len(batch)

end = time.time()

##############################################
# show resultes
##############################################
print("\n  processing done ")
print(f" time of processing : {end - start:.2f} seconds")
print(f"   the number of all sambels : {count:,}")
print(f"\n minimum value : {min_pred:.6f}\n")

for name, val in zip(cols, min_params):
    print(f"{name}: {val:.3f}")