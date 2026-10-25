import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI StudyMate", page_icon="📚")

st.title("AI StudyMate")
st.write("Student Performance Prediction Demo")

st.warning(
    "Educational demo only. This model uses sample data "
    "and is not validated for real student predictions."
)

data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 2, 4, 6, 3, 5],
    "attendance": [50, 60, 65, 70, 80, 90, 95, 55, 75, 85, 68, 88],
    "previous_score": [35, 40, 50, 55, 65, 75, 90, 42, 60, 80, 48, 70],
    "result": [0, 0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1]
})

X = data[["study_hours", "attendance", "previous_score"]]
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.subheader("Enter student details")

hours = st.slider("Study hours per day", 0, 12, 4)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.slider("Previous marks (%)", 0, 100, 60)

if st.button("Predict performance"):
    sample = pd.DataFrame([{
        "study_hours": hours,
        "attendance": attendance,
        "previous_score": previous_score
    }])

    prediction = int(model.predict(sample)[0])

    results = {
        0: "Needs Improvement",
        1: "Average Performance",
        2: "Strong Performance"
    }

    st.success("Estimated category: " + results[prediction])

st.metric("Demo model test accuracy", f"{accuracy:.0%}")
st.caption("Accuracy is based on a very small sample dataset.")
