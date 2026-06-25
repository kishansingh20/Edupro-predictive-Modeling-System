"""
=====================================================
EduPro - ML Pipeline (Updated)
=====================================================
Run karo:
    python ml_pipeline.py
=====================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("  EduPro ML Pipeline Starting...")
print("=" * 55)

# ─────────────────────────────────────────
# 1. DATA LOAD
# ─────────────────────────────────────────
df = pd.read_csv("master_data.csv")
print(f"\n✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n📐 Feature Engineering...")

df["PriceBand"] = pd.cut(
    df["CoursePrice"], bins=[-1, 0, 100, 300, 500],
    labels=["Free", "Low", "Medium", "High"]
)
df["DurationBucket"] = pd.cut(
    df["CourseDuration"], bins=[0, 10, 25, 50],
    labels=["Short", "Medium", "Long"]
)
df["RatingTier"] = pd.cut(
    df["CourseRating"], bins=[0, 2.5, 3.5, 4.2, 5],
    labels=["Low", "Medium", "High", "Top"]
)
df["ExperienceBucket"] = pd.cut(
    df["YearsOfExperience"], bins=[0, 5, 12, 25],
    labels=["Junior", "Mid", "Senior"]
)
df["ExpertiseMatch"] = (df["Expertise"] == df["CourseCategory"]).astype(int)
print("   ✅ PriceBand, DurationBucket, RatingTier, ExperienceBucket, ExpertiseMatch — done")

# ─────────────────────────────────────────
# 3. COURSE-LEVEL AGGREGATION
# ─────────────────────────────────────────
print("\n📊 Aggregating to course level...")

course_agg = df.groupby("CourseID").agg(
    EnrollmentCount   = ("TransactionID",    "count"),
    CourseRevenue     = ("Amount",           "sum"),
    CourseCategory    = ("CourseCategory",   "first"),
    CourseLevel       = ("CourseLevel",      "first"),
    CourseType        = ("CourseType",       "first"),
    CoursePrice       = ("CoursePrice",      "first"),
    CourseDuration    = ("CourseDuration",   "first"),
    CourseRating      = ("CourseRating",     "first"),
    TeacherRating     = ("TeacherRating",    "first"),
    YearsOfExperience = ("YearsOfExperience","first"),
    ExpertiseMatch    = ("ExpertiseMatch",   "first"),
    PriceBand         = ("PriceBand",        "first"),
    DurationBucket    = ("DurationBucket",   "first"),
    RatingTier        = ("RatingTier",       "first"),
    ExperienceBucket  = ("ExperienceBucket", "first"),
).reset_index()
print(f"   ✅ {course_agg.shape[0]} unique courses aggregated")

# ─────────────────────────────────────────
# 4. LABEL ENCODING
# ─────────────────────────────────────────
print("\n🔢 Label Encoding...")

CAT_COLS = [
    "CourseCategory", "CourseLevel", "CourseType",
    "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket"
]

encoders   = {}
course_enc = course_agg.copy()

for col in CAT_COLS:
    course_enc[col] = course_enc[col].astype(str)
    le = LabelEncoder()
    course_enc[col] = le.fit_transform(course_enc[col])
    encoders[col] = le

os.makedirs("models", exist_ok=True)
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# ─────────────────────────────────────────
# 5. CORRELATION CHECK & REMOVE REDUNDANT
# ─────────────────────────────────────────
print("\n🔍 Correlation Check (removing redundant features)...")

ALL_FEATURES = [
    "CourseCategory", "CourseLevel", "CourseType",
    "CoursePrice", "CourseDuration", "CourseRating",
    "TeacherRating", "YearsOfExperience", "ExpertiseMatch",
    "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket"
]

X_full = course_enc[ALL_FEATURES].copy()
corr_matrix = X_full.corr().abs()

# Find and drop highly correlated features (threshold = 0.85)
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
print(f"   Highly correlated features removed (>0.85): {to_drop}")

FEATURES = [f for f in ALL_FEATURES if f not in to_drop]
print(f"   ✅ Final features ({len(FEATURES)}): {FEATURES}")

# Save correlation matrix and dropped features
corr_info = {
    "correlation_matrix": corr_matrix,
    "dropped_features": to_drop,
    "final_features": FEATURES
}
with open("models/corr_info.pkl", "wb") as f:
    pickle.dump(corr_info, f)

with open("models/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

# ─────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X         = course_enc[FEATURES]
y_enroll  = course_enc["EnrollmentCount"]
y_revenue = course_enc["CourseRevenue"]

X_train, X_test, ye_train, ye_test = train_test_split(X, y_enroll,  test_size=0.2, random_state=42)
_,       _,      yr_train, yr_test = train_test_split(X, y_revenue, test_size=0.2, random_state=42)

# ─────────────────────────────────────────
# 7. MODEL DEFINITIONS
# ─────────────────────────────────────────
import copy

MODEL_DEFS = {
    "Linear Regression":  LinearRegression(),
    "Ridge Regression":   Ridge(alpha=1.0),
    "Lasso Regression":   Lasso(alpha=0.1),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

# ─────────────────────────────────────────
# 8. TRAIN & EVALUATE
# ─────────────────────────────────────────
def train_evaluate(model_defs, X_tr, X_te, y_tr, y_te):
    print(f"   {'Model':<26} {'MAE':>10} {'RMSE':>10} {'R2':>8}")
    print("   " + "-" * 58)
    results = {}
    trained = {}
    for name, mdl in model_defs.items():
        m = copy.deepcopy(mdl)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mae   = mean_absolute_error(y_te, preds)
        rmse  = np.sqrt(mean_squared_error(y_te, preds))
        r2    = r2_score(y_te, preds)
        results[name] = {"MAE": float(round(mae,2)), "RMSE": float(round(rmse,2)), "R2": float(round(r2,3))}
        trained[name] = m
        print(f"   {name:<26} {mae:>10.2f} {rmse:>10.2f} {r2:>8.3f}")
    return results, trained

print("\n" + "=" * 55)
print("  ENROLLMENT PREDICTION")
print("=" * 55)
enroll_results, enroll_models = train_evaluate(MODEL_DEFS, X_train, X_test, ye_train, ye_test)

print("\n" + "=" * 55)
print("  REVENUE PREDICTION")
print("=" * 55)
revenue_results, revenue_models = train_evaluate(MODEL_DEFS, X_train, X_test, yr_train, yr_test)

# ─────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────
print("\n📊 Extracting Feature Importance...")

feat_importance = {}

# Random Forest importances (enrollment)
rf_enroll = enroll_models["Random Forest"]
feat_importance["enroll_rf"] = dict(zip(FEATURES, rf_enroll.feature_importances_))

# Random Forest importances (revenue)
rf_revenue = revenue_models["Random Forest"]
feat_importance["revenue_rf"] = dict(zip(FEATURES, rf_revenue.feature_importances_))

# Gradient Boosting importances (enrollment)
gb_enroll = enroll_models["Gradient Boosting"]
feat_importance["enroll_gb"] = dict(zip(FEATURES, gb_enroll.feature_importances_))

# Gradient Boosting importances (revenue)
gb_revenue = revenue_models["Gradient Boosting"]
feat_importance["revenue_gb"] = dict(zip(FEATURES, gb_revenue.feature_importances_))

with open("models/feature_importance.pkl", "wb") as f:
    pickle.dump(feat_importance, f)

print("   ✅ Feature importance saved")

# Print top 5
top5_rev = sorted(feat_importance["revenue_rf"].items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\n   Top 5 Revenue Drivers:")
for feat, imp in top5_rev:
    print(f"      {feat:<25} {imp:.4f}")

# ─────────────────────────────────────────
# 10. SAVE ALL MODELS
# ─────────────────────────────────────────
best_enroll_name  = max(enroll_results,  key=lambda k: enroll_results[k]["R2"])
best_revenue_name = max(revenue_results, key=lambda k: revenue_results[k]["R2"])

with open("models/best_enroll_model.pkl",   "wb") as f: pickle.dump(enroll_models[best_enroll_name],   f)
with open("models/best_revenue_model.pkl",  "wb") as f: pickle.dump(revenue_models[best_revenue_name], f)
with open("models/all_enroll_models.pkl",   "wb") as f: pickle.dump(enroll_models,                     f)
with open("models/all_revenue_models.pkl",  "wb") as f: pickle.dump(revenue_models,                    f)
with open("models/enroll_results.pkl",      "wb") as f: pickle.dump(enroll_results,                    f)
with open("models/revenue_results.pkl",     "wb") as f: pickle.dump(revenue_results,                   f)

print("\n" + "=" * 55)
print(f"  ✅ Best Enrollment Model : {best_enroll_name}")
print(f"     R2 = {enroll_results[best_enroll_name]['R2']}")
print(f"  ✅ Best Revenue Model    : {best_revenue_name}")
print(f"     R2 = {revenue_results[best_revenue_name]['R2']}")
print("=" * 55)
print("\n  All models saved in /models/ folder")
print("  Now run:  streamlit run app.py")
print("=" * 55)
