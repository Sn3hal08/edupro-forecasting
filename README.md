# EduPro Course Demand and Revenue Forecasting

Streamlit app that predicts **enrollment count**, **course revenue** and **category revenue** from course and instructor attributes.

## Run
```bash
pip install -r requirements.txt
python generate_sample_data.py      # optional: creates SYNTHETIC demo data
streamlit run app.py
```
Upload your real workbook in the sidebar. It reads four sheets (Users, Teachers, Courses, Transactions). Column names are matched case-insensitively with common aliases (e.g. Category, Price, Date); Courses needs CourseID, CourseCategory and CoursePrice, and Transactions needs CourseID, TransactionDate and Amount. Other columns are optional and missing ones are reported in the sidebar. Expected columns:
- **Courses:** CourseID, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating, TeacherID
- **Teachers:** TeacherID, Expertise, YearsOfExperience, TeacherRating
- **Users:** UserID, Age, Gender (used for the data overview only)
- **Transactions:** TransactionID, UserID, CourseID, TransactionDate, Amount

If TeacherID is not in Courses but is in Transactions, the most frequent teacher per course is used.

## How it works
- `src/pipeline.py`: merge and aggregate to one row per course, engineer features (price bands, duration buckets, rating tiers, experience buckets, expertise-category match, history), drop correlated features, train Linear, Ridge, Lasso, Random Forest and Gradient Boosting, report MAE, RMSE and R2, and compute feature importance.
- Category revenue is the sum of out-of-fold course revenue predictions per category.
- **History toggle:** off, models predict for new courses from attributes alone. On, history comes from the first 70% of the timeline and targets from the last 30%, so history never leaks the target.
- `app.py`: the five dashboard tabs (what-if planner, revenue forecast, feature importance, category comparison, model performance).

The bundled `data/EduPro_Sample_Data.xlsx` is synthetic and only for demonstration. Do not report its numbers as findings.
