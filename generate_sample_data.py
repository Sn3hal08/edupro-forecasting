"""Creates a SYNTHETIC EduPro-style workbook (Courses, Teachers, Transactions) for demo/testing only."""
from pathlib import Path
import numpy as np
import pandas as pd

CATS = {"Data Science": 1.4, "Web Development": 1.2, "Design": 0.9, "Business": 0.8, "Marketing": 0.7, "Photography": 0.5}
LEVELS = {"Beginner": 1.3, "Intermediate": 1.0, "Advanced": 0.7}
TYPES = ["Self-paced", "Live", "Bootcamp"]


def make_workbook(path, seed=7, n_courses=120, n_teachers=30):
    rng = np.random.default_rng(seed)
    cats = list(CATS)
    teachers = pd.DataFrame({
        "TeacherID": [f"T{i:03d}" for i in range(n_teachers)],
        "Expertise": rng.choice(cats, n_teachers),
        "YearsOfExperience": rng.integers(1, 25, n_teachers),
        "TeacherRating": rng.uniform(3.2, 5.0, n_teachers).round(2),
    })
    cat = rng.choice(cats, n_courses)
    level = rng.choice(list(LEVELS), n_courses, p=[0.4, 0.4, 0.2])
    base = {"Beginner": 40, "Intermediate": 90, "Advanced": 150}
    courses = pd.DataFrame({
        "CourseID": [f"C{i:03d}" for i in range(n_courses)],
        "CourseCategory": cat,
        "CourseType": rng.choice(TYPES, n_courses),
        "CourseLevel": level,
        "CoursePrice": [round(base[l] * rng.uniform(0.6, 1.8), 2) for l in level],
        "CourseDuration": rng.integers(4, 60, n_courses),
        "CourseRating": rng.uniform(3.0, 5.0, n_courses).round(2),
        "TeacherID": rng.choice(teachers.TeacherID, n_courses),
    })
    t = courses.merge(teachers, on="TeacherID")
    lam = (60 * t.CourseCategory.map(CATS) * t.CourseLevel.map(LEVELS) * np.exp(-t.CoursePrice / 220)
           * (t.CourseRating / 4) ** 2 * (0.7 + t.TeacherRating / 10))
    counts = rng.poisson(lam)
    start, days = pd.Timestamp("2024-01-01"), 730
    rows = []
    for cid, price, n in zip(courses.CourseID, courses.CoursePrice, counts):
        for _ in range(n):
            rows.append((cid, start + pd.Timedelta(days=int(rng.beta(1.4, 1) * days)),
                         round(price * rng.choice([1, 1, 1, 0.9, 0.8]), 2)))
    users = pd.DataFrame({"UserID": [f"U{i:05d}" for i in range(2000)], "UserName": [f"learner{i}" for i in range(2000)],
                          "Age": rng.integers(15, 40, 2000), "Gender": rng.choice(["Male", "Female"], 2000)})
    users["Email"] = users.UserName + "@example.com"
    tx = pd.DataFrame(rows, columns=["CourseID", "TransactionDate", "Amount"])
    tx.insert(0, "UserID", rng.choice(users.UserID, len(tx)))
    tx.insert(0, "TransactionID", [f"TX{i:06d}" for i in range(len(tx))])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as w:
        users.to_excel(w, sheet_name="Users", index=False)
        courses.to_excel(w, sheet_name="Courses", index=False)
        teachers.to_excel(w, sheet_name="Teachers", index=False)
        tx.to_excel(w, sheet_name="Transactions", index=False)
    return path


if __name__ == "__main__":
    print("Wrote", make_workbook(Path(__file__).parent / "data" / "EduPro_Sample_Data.xlsx"))
