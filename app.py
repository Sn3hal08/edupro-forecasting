from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.pipeline import feature_importance, predict_one, run_all

st.set_page_config(page_title="EduPro Forecasting", page_icon="📈", layout="wide")
HERE = Path(__file__).parent


@st.cache_resource(show_spinner="Training models...")
def train(data, use_history, test_size):
    return run_all(data, use_history, test_size)


st.title("📈 EduPro: Course Demand and Revenue Forecasting")
st.caption("Predict enrollments and revenue from course and instructor attributes.")

with st.sidebar:
    st.header("Data and model")
    up = st.file_uploader("Upload EduPro workbook (.xlsx)", type="xlsx",
                          help="Sheets: Courses, Teachers, Transactions")
    use_hist = st.toggle("Use historical performance features", value=False,
                         help="Off: forecasts for brand-new courses. On: history from the first 70% of the timeline "
                              "predicts the last 30%.")
    test_size = st.slider("Test set share", 0.1, 0.4, 0.2, 0.05)

if up:
    data = up.getvalue()
else:
    sample = HERE / "data" / "EduPro_Sample_Data.xlsx"
    if not sample.exists():
        from generate_sample_data import make_workbook
        make_workbook(sample)
    data = sample.read_bytes()
    st.sidebar.warning("Showing SYNTHETIC demo data. Upload your EduPro workbook for real results.")

try:
    R = train(data, use_hist, test_size)
except ValueError as e:
    st.error(str(e))
    st.stop()

for w in R["warnings"]:
    st.sidebar.info(w)
df, pred = R["df"], R["pred"]
me, mr = R["enr"]["best"], R["rev"]["best"]
t1, t2, t3, t4, t5, t6 = st.tabs(["Demand prediction", "Revenue forecast", "Feature importance",
                                  "Category comparison", "Model performance", "Data overview"])

with t1:
    st.subheader("What-if course planner")
    a, b, c = st.columns(3)
    cat = a.selectbox("Category", sorted(df.CourseCategory.dropna().unique()))
    ctype = a.selectbox("Course type", sorted(df.CourseType.dropna().unique()))
    level = a.selectbox("Level", sorted(df.CourseLevel.dropna().unique()))
    price = b.number_input("Course price", 0.0, float(df.CoursePrice.max() * 3), float(df.CoursePrice.median()))
    dur = b.number_input("Duration", 0.0, float(df.CourseDuration.max() * 3), float(df.CourseDuration.median()))
    rating = b.slider("Course rating", 1.0, 5.0, float(min(5.0, max(1.0, round(df.CourseRating.median(), 1)))), 0.1)
    years = c.slider("Instructor experience (years)", 0, 40, int(df.YearsOfExperience.median()))
    trat = c.slider("Instructor rating", 1.0, 5.0, float(min(5.0, max(1.0, round(df.TeacherRating.median(), 1)))), 0.1)
    opts = sorted(df.Expertise.dropna().unique())
    expertise = c.selectbox("Instructor expertise", opts, index=opts.index(cat) if cat in opts else 0)
    row = {"CourseCategory": cat, "CourseType": ctype, "CourseLevel": level, "CoursePrice": price,
           "CourseDuration": dur, "CourseRating": rating, "Expertise": expertise, "YearsOfExperience": years,
           "TeacherRating": trat, "PastEnrollments": 0, "PastAvgRevenue": 0, "RevPerEnrollment": 0}
    if R["use_history"]:
        with st.expander("Historical performance inputs", expanded=True):
            row["PastEnrollments"] = st.number_input("Past enrollments", 0.0, value=float(df.PastEnrollments.median()))
            row["PastAvgRevenue"] = st.number_input("Past average monthly revenue", 0.0, value=float(df.PastAvgRevenue.median()))
            row["RevPerEnrollment"] = st.number_input("Revenue per enrollment", 0.0, value=float(df.RevPerEnrollment.median()))
    me_pipe, mr_pipe = R["enr"]["models"][me], R["rev"]["models"][mr]
    e = predict_one(me_pipe, R["bins"], row, R["num"], R["cat"])
    r = predict_one(mr_pipe, R["bins"], row, R["num"], R["cat"])
    m1, m2 = st.columns(2)
    m1.metric("Predicted enrollments", f"{e:,.0f}")
    m2.metric("Predicted revenue", f"{r:,.0f}")
    st.caption(f"Models: {me} (enrollments), {mr} (revenue). Estimates are associations learned from past data, not guarantees.")
    prices = np.linspace(max(price * 0.5, 1), max(price * 1.5, 2), 15)
    curve = pd.DataFrame({"Price": prices,
                          "Enrollments": [predict_one(me_pipe, R["bins"], {**row, "CoursePrice": p}, R["num"], R["cat"]) for p in prices],
                          "Revenue": [predict_one(mr_pipe, R["bins"], {**row, "CoursePrice": p}, R["num"], R["cat"]) for p in prices]})
    l, r_ = st.columns(2)
    l.plotly_chart(px.line(curve, x="Price", y="Enrollments", title="Price sensitivity: enrollments"), width="stretch")
    r_.plotly_chart(px.line(curve, x="Price", y="Revenue", title="Price sensitivity: revenue"), width="stretch")

with t2:
    st.subheader("Revenue forecast")
    st.caption("Predictions are out-of-fold (5-fold cross-validation), so no course is predicted by a model that saw it.")
    cs = R["categories"].melt("CourseCategory", ["ActualRevenue", "PredRevenue"], "Series", "Revenue")
    st.plotly_chart(px.bar(cs, x="CourseCategory", y="Revenue", color="Series", barmode="group",
                           title="Category revenue: actual vs predicted"), width="stretch")
    sc = px.scatter(pred, x="Revenue", y="PredRevenue", color="CourseCategory",
                    hover_data=["CourseID"], title="Course revenue: actual vs predicted")
    top = float(max(pred.Revenue.max(), pred.PredRevenue.max()))
    sc.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash"))
    st.plotly_chart(sc, width="stretch")

with t3:
    st.subheader("Feature importance explorer")
    a, b, c = st.columns(3)
    target = a.selectbox("Target", ["Enrollments", "Revenue"])
    store = R["enr"] if target == "Enrollments" else R["rev"]
    name = b.selectbox("Model", list(store["models"]), index=list(store["models"]).index(store["best"]))
    top = c.slider("Top features", 5, 25, 12)
    imp = feature_importance(store["models"][name], top).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "Relative importance", "index": "Feature"},
                           title=f"{name}: drivers of {target.lower()}"), width="stretch")
    st.caption("Importance shows what the model relies on, not that changing a feature will cause demand to change. "
               "For linear models it is the absolute coefficient on scaled features.")

with t4:
    st.subheader("Category-level comparison")
    cats = st.multiselect("Categories", sorted(df.CourseCategory.unique()), default=sorted(df.CourseCategory.unique()))
    cs = R["categories"][R["categories"].CourseCategory.isin(cats)]
    metric = st.radio("Show", ["AvgPredEnrollments", "AvgPredRevenue", "PredEnrollments", "PredRevenue"], horizontal=True)
    st.plotly_chart(px.bar(cs.sort_values(metric), x=metric, y="CourseCategory", orientation="h"), width="stretch")
    st.dataframe(cs.round(1), width="stretch", hide_index=True)

with t5:
    st.subheader("Model performance (held-out test set)")
    for title, store in [("Enrollment count", R["enr"]), ("Course revenue", R["rev"])]:
        st.markdown(f"**{title}** (best: {store['best']})")
        st.dataframe(store["results"].round(3), width="stretch", hide_index=True)
    if R["dropped"]:
        st.info(f"Removed as highly correlated: {', '.join(R['dropped'])}")
    with st.expander("Modeling table preview"):
        st.dataframe(df.head(50), width="stretch")

with t6:
    st.subheader("Data overview")
    o = R["overview"]
    k = st.columns(5)
    k[0].metric("Courses", f"{o['courses']:,}")
    k[1].metric("Teachers", f"{o['teachers']:,}")
    k[2].metric("Learners", f"{o['learners']:,}")
    k[3].metric("Transactions", f"{o['transactions']:,}")
    k[4].metric("Period", o["period"])
    st.caption(f"Total revenue in the data: {o['revenue']:,.0f}")
    u = R["users"]
    if len(u) and u.Age.notna().any():
        l, r_ = st.columns(2)
        l.plotly_chart(px.histogram(u, x="Age", nbins=20, title="Learner age distribution"), width="stretch")
        g = u.Gender.fillna("Unknown").value_counts().reset_index()
        r_.plotly_chart(px.pie(g, names="Gender", values="count", title="Learner gender mix"), width="stretch")
    else:
        st.info("No usable Users sheet found, so learner demographics are skipped.")
    top = pred.groupby("CourseCategory")[["Enrollments", "Revenue"]].sum().reset_index()
    st.plotly_chart(px.bar(top.sort_values("Revenue"), x="Revenue", y="CourseCategory", orientation="h",
                           title="Actual revenue by category"), width="stretch")
