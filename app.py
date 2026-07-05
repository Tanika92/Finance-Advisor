import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="AI Personal Finance Advisor",
    page_icon="💰",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#f7fbff,#eef7ff,#f8fffb);
}

/* Hero Banner */
.hero{
    background: linear-gradient(135deg,#0f766e,#22c55e,#4ade80);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 10px 30px rgba(0,0,0,0.25);
    margin-bottom:30px;
}

.hero h1{
    font-size:48px;
    margin-bottom:8px;
    font-weight:700;
}

.hero p{
    font-size:22px;
    opacity:0.95;
}

/* Cards */
div[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.15);
}

/* Button */
.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background:#16a34a;
    color:white;
    border:none;
}

.stButton>button:hover{
    background:#15803d;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#eef5ff,#f9fcff,#eef7f6);
}

/* Main container */
.block-container{
    background:white;
    border-radius:20px;
    padding:2rem;
    box-shadow:0px 8px 30px rgba(0,0,0,0.08);
}

/* Headings */
h1,h2,h3{
    color:#0f172a;
    font-family:Arial;
    font-weight:700;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:18px;
    box-shadow:0 6px 18px rgba(0,0,0,0.08);
    border-left:6px solid #2E8B57;
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    border-radius:12px !important;
}

/* Button */
.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#2E8B57,#4CAF50);
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#4CAF50,#2E8B57);
}

/* Charts */
canvas{
    background:white;
    border-radius:18px;
}

/* Success */
div[data-testid="stAlert"]{
    border-radius:15px;
}

/* Progress bar */
.stProgress > div > div{
    background:#2E8B57;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style='background:linear-gradient(90deg,#2E8B57,#66BB6A);
padding:18px;
border-radius:18px;
text-align:center;
color:white;
margin-bottom:20px;'>

<h1>💰 AI Personal Finance Advisor</h1>

<h4>Predict expenses, analyse savings and receive AI-powered financial advice.</h4>

</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Personal Finance Advisor",
    page_icon="💰",
    layout="wide"
)
st.divider()

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

df = pd.read_csv("personal_expense_dataset.csv")

# -------------------------------------------------
# ENCODING
# -------------------------------------------------

category_encoder = LabelEncoder()
payment_encoder = LabelEncoder()
month_encoder = LabelEncoder()

df["Category"] = category_encoder.fit_transform(df["Category"])
df["Payment_Mode"] = payment_encoder.fit_transform(df["Payment_Mode"])
df["Month"] = month_encoder.fit_transform(df["Month"])

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

X = df[["Category","Payment_Mode","Month"]]
y = df["Amount"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

# -------------------------------------------------
# USER DETAILS
# -------------------------------------------------

st.header("👤 User Information")

col1,col2=st.columns(2)

with col1:

    name=st.text_input("Full Name")

    age=st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=21
    )

    occupation=st.selectbox(
        "Occupation",
        [
            "Student",
            "Engineer",
            "Business",
            "Teacher",
            "Doctor",
            "Other"
        ]
    )

    city=st.text_input("City")

with col2:

    income=st.number_input(
        "Monthly Income (₹)",
        min_value=0.0,
        step=1000.0
    )

    category=st.selectbox(
        "Expense Category",
        category_encoder.classes_
    )

    payment_mode=st.selectbox(
        "Payment Mode",
        payment_encoder.classes_
    )

    month=st.selectbox(
        "Month",
        month_encoder.classes_
    )

st.divider()

st.subheader("🤖 AI Expense Prediction")

predict = st.button(
    "Predict My Monthly Expense",
    use_container_width=True
)
# ==========================================================
# AI PREDICTION
# ==========================================================

if predict:

    # Encode user inputs
    category_encoded = category_encoder.transform([category])[0]
    payment_encoded = payment_encoder.transform([payment_mode])[0]
    month_encoded = month_encoder.transform([month])[0]

    input_data = pd.DataFrame(
        [[category_encoded, payment_encoded, month_encoded]],
        columns=["Category", "Payment_Mode", "Month"]
    )

    # AI Prediction
    prediction = model.predict(input_data)[0]

    # Savings
    saving = income - prediction

    st.divider()

    st.header("🤖 AI Financial Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "💸 Predicted Expense",
            f"₹ {prediction:,.2f}"
        )

    with c2:
        st.metric(
            "💰 Estimated Savings",
            f"₹ {saving:,.2f}"
        )

    st.divider()

    # Financial Health

    health = (saving / income) * 100 if income > 0 else 0

    if health >= 40:
        st.success("🟢 Excellent Financial Health")

    elif health >= 20:
        st.warning("🟡 Good Financial Health")

    else:
        st.error("🔴 Poor Financial Health")

    # Spending Level

    if prediction < income * 0.50:
      spending="Low"

    elif prediction < income * 0.80:
      spending="Moderate"

    else:
       spending="High"
    st.metric(
    "📊 Spending Level",
    spending
)

    st.divider()

    st.subheader("📊 Expense Analysis Dashboard")

    # -----------------------------
    # PIE CHART
    # -----------------------------

    fig1, ax1 = plt.subplots(figsize=(4,4))

    ax1.pie(
        [prediction, max(saving,0)],
        labels=["Expense","Savings"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05,0)
    )

    ax1.set_title("Expense vs Savings")

    st.pyplot(fig1)

    # -----------------------------
    # BAR GRAPH
    # -----------------------------

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.bar(
        ["Income","Expense","Savings"],
        [income,prediction,max(saving,0)]
    )

    ax2.set_ylabel("Amount (₹)")
    ax2.set_title("Monthly Financial Summary")

    st.pyplot(fig2)

    # -----------------------------
    # LINE GRAPH
    # -----------------------------

    months = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    trend = np.linspace(
        prediction*0.75,
        prediction*1.10,
        12
    )

    fig3, ax3 = plt.subplots(figsize=(9,4))

    ax3.plot(
        months,
        trend,
        marker="o",
        linewidth=3
    )

    ax3.set_title("Predicted Expense Trend")

    ax3.set_ylabel("Expense (₹)")

    st.pyplot(fig3)

    st.divider()

    st.header("💡 AI Financial Advice")

    if category=="Food":
        st.info("🍲 Cook more meals at home to reduce food expenses.")

    elif category=="Shopping":
        st.info("🛍 Avoid impulse buying and prepare a shopping list before purchasing.")

    elif category=="Entertainment":
        st.info("🎬 Set a monthly entertainment budget.")

    elif category=="Travel":
        st.info("✈ Use public transport whenever possible.")

    elif category=="Bills":
        st.info("💡 Pay bills before the due date to avoid penalties.")

    else:
        st.info("📌 Track your monthly expenses regularly.")

    if saving < income*0.20:
        st.warning(
            "You should try saving at least 20% of your monthly income."
        )

    else:
        st.success(
            "Excellent! Your savings habit is healthy."
        )

    st.divider()

    st.header("📈 Future Prediction")

    future = saving * 12

    st.metric(
        "Estimated Savings After 1 Year",
        f"₹ {future:,.2f}"
    )

    progress = min(max(health,0),100)

    st.progress(progress/100)

    st.write(f"Financial Health Score : **{progress:.1f}/100**")

    st.divider()

    st.header("📄 Project Summary")

    st.write("✅ Model : Random Forest Regressor")
    st.write("✅ Features : Category, Payment Mode, Month")
    st.write("✅ Prediction : Monthly Expense")
    st.write("✅ Personalized Financial Advice")
    st.write("✅ AI Dashboard with Interactive Charts")
    st.success("Thank you for using AI Personal Finance Advisor!")
    