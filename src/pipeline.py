"""EduPro demand and revenue forecasting: data prep, features, models, evaluation."""
import io
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUM = ["CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating", "ExpertiseMatch"]
HIST = ["PastEnrollments", "PastAvgRevenue", "RevPerEnrollment"]
CAT = ["CourseCategory", "CourseType", "CourseLevel", "PriceBand", "DurationBucket", "RatingTier", "ExpBucket"]
ALIASES = {
    "Courses": {"CourseID": ["courseid", "id"], "CourseCategory": ["coursecategory", "category"],
                "CourseType": ["coursetype", "type"], "CourseLevel": ["courselevel", "level"],
                "CoursePrice": ["courseprice", "price"], "CourseDuration": ["courseduration", "duration"],
                "CourseRating": ["courserating", "rating"], "TeacherID": ["teacherid", "instructorid"]},
    "Teachers": {"TeacherID": ["teacherid", "instructorid", "id"], "Expertise": ["expertise", "specialization", "speciality"],
                 "YearsOfExperience": ["yearsofexperience", "experience", "experienceyears"],
                 "TeacherRating": ["teacherrating", "rating"]},
    "Transactions": {"TransactionID": ["transactionid", "id"], "CourseID": ["courseid"], "UserID": ["userid"],
                     "TeacherID": ["teacherid"], "TransactionDate": ["transactiondate", "date"],
                     "Amount": ["amount", "revenue", "price"]},
    "Users": {"UserID": ["userid", "id"], "Age": ["age"], "Gender": ["gender"]},
}
CORE = {"Courses": ["CourseID", "CourseCategory", "CoursePrice"], "Teachers": [],
        "Transactions": ["CourseID", "TransactionDate", "Amount"], "Users": []}
NUMERIC = {"CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience", "TeacherRating", "Amount", "Age"}
IDS = {"CourseID", "TeacherID", "UserID", "TransactionID"}


def _num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def _standardize(df, sheet, warns):
    """Map whatever column names the sheet uses onto the canonical names (case/spacing-insensitive)."""
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, aliases in ALIASES[sheet].items():
        src = next((norm[a] for a in aliases if a in norm), None) or \
            next((norm[k] for k in norm for a in aliases if len(a) >= 5 and k.startswith(a)), None)
        if src is not None:
            col = df[src]
            out[canon] = _num(col) if canon in NUMERIC else (col.astype(str).str.strip().where(col.notna()) if canon in IDS else col)
        elif canon in CORE[sheet]:
            raise ValueError(f"Sheet '{sheet}' has no column matching '{canon}'. Columns found: {list(df.columns)}")
        elif canon == "TransactionID":
            out[canon] = np.arange(len(df)).astype(str)
        elif canon in IDS:
            out[canon] = pd.Series([None] * len(df), index=df.index, dtype=object)
        else:
            warns.append(f"'{sheet}' has no {canon} column; treated as {'0' if canon in NUMERIC else 'Unknown'}.")
            out[canon] = 0.0 if canon in NUMERIC else "Unknown"
        if canon in NUMERIC and out[canon].isna().all():
            out[canon] = 0.0
        if canon not in NUMERIC and canon not in IDS:
            out[canon] = out[canon].fillna("Unknown").astype(str)
    return out


def load_workbook(src):
    """Read the EduPro workbook (Users, Teachers, Courses, Transactions sheets) tolerantly."""
    raw = {k.strip().lower(): v for k, v in pd.read_excel(src, sheet_name=None).items()}
    warns, out = [], {}
    for sheet in ["Courses", "Transactions", "Teachers", "Users"]:
        key = next((k for k in (sheet.lower(), sheet.lower().rstrip("s")) if k in raw), None)
        if key is None:
            if CORE[sheet]:
                raise ValueError(f"Workbook needs a '{sheet}' sheet. Sheets found: {list(raw)}")
            warns.append(f"No '{sheet}' sheet found; continuing without it.")
            raw[sheet.lower()] = pd.DataFrame(columns=list(ALIASES[sheet]))
            key = sheet.lower()
        out[sheet] = _standardize(raw[key], sheet, warns)
    out["Courses"] = out["Courses"].drop_duplicates("CourseID")
    out["Teachers"] = out["Teachers"].drop_duplicates("TeacherID")
    tx = out["Transactions"]
    tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
    if tx["TransactionDate"].isna().all():
        raise ValueError("No valid dates found in Transactions.TransactionDate.")
    return {"courses": out["Courses"], "teachers": out["Teachers"], "tx": tx, "users": out["Users"], "warnings": warns}


def _bucket(x, edges, labels):
    x = x.fillna(x.median())
    return pd.Series(np.array(labels, dtype=object)[np.digitize(x, edges)], index=x.index)


def _match(expertise, category):
    e, c = str(expertise).lower().strip(), str(category).lower().strip()
    if e == c:
        return 1.0
    return 0.5 if set(e.split()) & set(c.split()) else 0.0


def add_features(df, bins):
    df = df.copy()
    df["PriceBand"] = _bucket(df.CoursePrice, bins["price"], ["low", "medium", "high"])
    df["DurationBucket"] = _bucket(df.CourseDuration, bins["duration"], ["short", "medium", "long"])
    df["RatingTier"] = _bucket(df.CourseRating, [3.5, 4.2], ["low", "good", "excellent"])
    df["ExpBucket"] = _bucket(df.YearsOfExperience, [2, 5, 10], ["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
    df["ExpertiseMatch"] = [_match(a, b) for a, b in zip(df.Expertise, df.CourseCategory)]
    return df


def build_course_table(courses, teachers, tx, use_history=False):
    """One row per course. With use_history, history comes from the first 70% of the
    timeline and targets from the remaining 30%, so features never see the target period."""
    tx = tx.copy()
    tx["TransactionDate"] = pd.to_datetime(tx.TransactionDate, errors="coerce")
    tx = tx.dropna(subset=["CourseID", "TransactionDate"])
    if use_history:
        cutoff = tx.TransactionDate.quantile(0.7)
        hist, fut = tx[tx.TransactionDate <= cutoff], tx[tx.TransactionDate > cutoff]
        months = max((cutoff - tx.TransactionDate.min()).days / 30.0, 1.0)
    else:
        hist, fut, months = tx.iloc[0:0], tx, 1.0
    tgt = fut.groupby("CourseID").agg(Enrollments=("TransactionID", "count"), Revenue=("Amount", "sum")).reset_index()
    h = hist.groupby("CourseID").agg(PastEnrollments=("TransactionID", "count"), PastRevenue=("Amount", "sum")).reset_index()
    df = courses.merge(tgt, on="CourseID", how="left").merge(h, on="CourseID", how="left")
    for c in ["Enrollments", "Revenue", "PastEnrollments", "PastRevenue"]:
        df[c] = df[c].fillna(0)
    df["PastAvgRevenue"] = df.PastRevenue / months
    df["RevPerEnrollment"] = (df.PastRevenue / df.PastEnrollments.replace(0, np.nan)).fillna(0)
    if df["TeacherID"].isna().all() and tx["TeacherID"].notna().any():
        main = tx.groupby("CourseID").TeacherID.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
        df = df.drop(columns="TeacherID").merge(main.reset_index(), on="CourseID", how="left")
    df = df.merge(teachers[["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]], on="TeacherID", how="left")
    for c in ["YearsOfExperience", "TeacherRating"]:
        if df[c].isna().all():
            df[c] = 0.0
    df["Expertise"] = df["Expertise"].fillna("Unknown")
    bins = {"price": np.quantile(df.CoursePrice.dropna(), [1 / 3, 2 / 3]),
            "duration": np.quantile(df.CourseDuration.dropna(), [1 / 3, 2 / 3])}
    return add_features(df, bins), bins


def drop_correlated(df, cols, thr=0.95):
    corr, drop = df[cols].corr().abs(), set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a not in drop and b not in drop and corr.loc[a, b] > thr:
                drop.add(b)
    return [c for c in cols if c not in drop], sorted(drop)


def make_pipeline(model, num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    return Pipeline([("pre", pre), ("model", model)])


def model_zoo():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-2, 3, 20)),
        "Lasso": LassoCV(cv=5, max_iter=50000),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def train_target(df, target, num, cat, test_size=0.2, seed=42):
    X, y = df[num + cat], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    rows, fitted = [], {}
    for name, m in model_zoo().items():
        pipe = make_pipeline(m, num, cat).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        rows.append({"Model": name, "MAE": mean_absolute_error(yte, p),
                     "RMSE": mean_squared_error(yte, p) ** 0.5, "R2": r2_score(yte, p)})
        fitted[name] = pipe
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = res.loc[0, "Model"]
    oof = cross_val_predict(make_pipeline(model_zoo()[best], num, cat), X, y, cv=5)
    return {"results": res, "models": fitted, "best": best, "oof": pd.Series(np.clip(oof, 0, None), index=df.index)}


def feature_importance(pipe, top=15):
    names = pipe.named_steps["pre"].get_feature_names_out()
    m = pipe.named_steps["model"]
    imp = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_)
    s = pd.Series(imp, index=[n.split("__", 1)[1] for n in names])
    return (s / (s.sum() or 1)).sort_values(ascending=False).head(top)


def category_summary(pred):
    g = pred.groupby("CourseCategory").agg(
        Courses=("CourseID", "size"), ActualEnrollments=("Enrollments", "sum"), PredEnrollments=("PredEnrollments", "sum"),
        ActualRevenue=("Revenue", "sum"), PredRevenue=("PredRevenue", "sum")).reset_index()
    g["AvgPredEnrollments"] = g.PredEnrollments / g.Courses
    g["AvgPredRevenue"] = g.PredRevenue / g.Courses
    return g


def predict_one(pipe, bins, row, num, cat):
    d = add_features(pd.DataFrame([row]), bins)
    return float(max(0.0, pipe.predict(d[num + cat])[0]))


def run_all(data: bytes, use_history=False, test_size=0.2):
    L = load_workbook(io.BytesIO(data))
    df, bins = build_course_table(L["courses"], L["teachers"], L["tx"], use_history)
    num = BASE_NUM + (HIST if use_history else [])
    num, dropped = drop_correlated(df, num)
    enr = train_target(df, "Enrollments", num, CAT, test_size)
    rev = train_target(df, "Revenue", num, CAT, test_size)
    pred = df.copy()
    pred["PredEnrollments"], pred["PredRevenue"] = enr["oof"], rev["oof"]
    return {"df": df, "pred": pred, "bins": bins, "num": num, "cat": CAT, "dropped": dropped,
            "enr": enr, "rev": rev, "categories": category_summary(pred), "use_history": use_history,
            "users": L["users"], "warnings": L["warnings"],
            "overview": {"courses": len(L["courses"]), "teachers": len(L["teachers"]), "learners": len(L["users"]),
                         "transactions": len(L["tx"]), "revenue": float(L["tx"].Amount.sum()),
                         "period": f"{L['tx'].TransactionDate.min():%b %Y} to {L['tx'].TransactionDate.max():%b %Y}"}}
