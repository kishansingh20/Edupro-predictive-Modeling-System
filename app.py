"""
EduPro Analytics Platform — app.py (Updated with all requirements)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Auto-train models if not already present (safe — no subprocess)
if not os.path.exists("models/best_enroll_model.pkl"):
    with st.spinner("⚙️ Training ML models for the first time... please wait."):
        import ml_pipeline  # runs the pipeline inline

st.set_page_config(page_title="EduPro Analytics Platform", page_icon="📊", layout="wide")

# ─── LOAD DATA ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("master_data.csv")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    return df

df_raw = load_data()

# ─── LOAD MODELS ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    mdir = "models"
    if not os.path.exists(mdir):
        return None
    try:
        def _load(name):
            with open(f"{mdir}/{name}", "rb") as f:
                return pickle.load(f)
        return {
            "encoders":        _load("encoders.pkl"),
            "enroll_model":    _load("best_enroll_model.pkl"),
            "revenue_model":   _load("best_revenue_model.pkl"),
            "all_enroll":      _load("all_enroll_models.pkl"),
            "all_revenue":     _load("all_revenue_models.pkl"),
            "enroll_results":  _load("enroll_results.pkl"),
            "revenue_results": _load("revenue_results.pkl"),
            "features":        _load("features.pkl"),
            "feat_importance": _load("feature_importance.pkl"),
            "corr_info":       _load("corr_info.pkl"),
        }
    except Exception as e:
        st.warning(f"Model load error: {e}")
        return None

ml = load_models()

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=70)
    st.title("EduPro")
    st.caption("Analytics & Forecasting Platform")
    st.divider()

    menu = st.radio("Navigation", [
        "📊 Overview",
        "👨‍🏫 Instructor Analysis",
        "📚 Course Quality",
        "🏆 Leaderboard",
        "🔍 Explorer",
        "🤖 Predictions",
        "📈 Feature Importance",
        "📂 Category Demand"
    ], label_visibility="collapsed")

    st.divider()
    st.subheader("Filters")
    category_filter = st.multiselect("Course Category",
        options=sorted(df_raw["CourseCategory"].unique()),
        default=sorted(df_raw["CourseCategory"].unique()))
    level_filter = st.multiselect("Course Level",
        options=sorted(df_raw["CourseLevel"].unique()),
        default=sorted(df_raw["CourseLevel"].unique()))
    type_filter = st.multiselect("Course Type",
        options=sorted(df_raw["CourseType"].unique()),
        default=sorted(df_raw["CourseType"].unique()))

df = df_raw[
    (df_raw["CourseCategory"].isin(category_filter)) &
    (df_raw["CourseLevel"].isin(level_filter)) &
    (df_raw["CourseType"].isin(type_filter))
].copy()


# ════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════
if menu == "📊 Overview":
    st.title("📊 EduPro Analytics Platform")
    st.markdown("#### Demand & Revenue Forecast Dashboard")
    st.caption(f"Showing {len(df):,} filtered records out of {len(df_raw):,} total")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue",   f"${df['Amount'].sum():,.0f}")
    c2.metric("🎓 Enrollments",     f"{len(df):,}")
    c3.metric("📖 Courses",         df["CourseID"].nunique())
    c4.metric("👨‍🏫 Teachers",       df["TeacherID"].nunique())
    c5.metric("⭐ Avg Rating",       f"{df['CourseRating'].mean():.2f}")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Revenue by Category")
        cat_rev = df.groupby("CourseCategory")["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
        fig = px.bar(cat_rev, x="CourseCategory", y="Amount", color="Amount",
                     color_continuous_scale="Blues", title="Revenue by Course Category")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Enrollments by Category")
        cat_enroll = df.groupby("CourseCategory").size().reset_index(name="Enrollments").sort_values("Enrollments", ascending=False)
        fig2 = px.pie(cat_enroll, names="CourseCategory", values="Enrollments",
                      hole=0.45, title="Enrollment Share by Category")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Revenue Trend")
    monthly = df.groupby("Month")["Amount"].sum().reset_index().sort_values("Month")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["MonthName"] = monthly["Month"].map(month_names)
    fig3 = px.line(monthly, x="MonthName", y="Amount", markers=True, title="Monthly Revenue Trend")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ════════════════════════════════════════════
# PAGE 2: INSTRUCTOR ANALYSIS
# ════════════════════════════════════════════
elif menu == "👨‍🏫 Instructor Analysis":
    st.title("👨‍🏫 Instructor Analysis")
    st.divider()

    teacher_summary = (
        df.groupby(["TeacherName","TeacherRating","YearsOfExperience","Expertise"])
        .agg(Enrollments=("TransactionID","count"), Revenue=("Amount","sum"))
        .reset_index().sort_values("Enrollments", ascending=False)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Instructors",     df["TeacherID"].nunique())
    c2.metric("Avg Instructor Rating", f"{df['TeacherRating'].mean():.2f}")
    c3.metric("Avg Experience",        f"{df['YearsOfExperience'].mean():.1f} yrs")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Top 10 Instructors by Enrollment")
        fig = px.bar(teacher_summary.head(10), x="TeacherName", y="Enrollments",
                     color="TeacherRating", color_continuous_scale="RdYlGn",
                     title="Top 10 Instructors by Enrollment")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Top 10 Instructors by Revenue")
        fig2 = px.bar(teacher_summary.sort_values("Revenue", ascending=False).head(10),
                      x="TeacherName", y="Revenue", color="Revenue",
                      color_continuous_scale="Viridis", title="Top 10 Instructors by Revenue")
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Rating vs Experience")
    fig3 = px.scatter(teacher_summary, x="YearsOfExperience", y="TeacherRating",
                      size="Enrollments", color="Expertise", hover_data=["TeacherName"],
                      title="Instructor Rating vs Experience (bubble = enrollments)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("All Instructors Data")
    st.dataframe(teacher_summary, use_container_width=True)


# ════════════════════════════════════════════
# PAGE 3: COURSE QUALITY
# ════════════════════════════════════════════
elif menu == "📚 Course Quality":
    st.title("📚 Course Quality")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Top 10 Rated Courses")
        top_courses = (df.groupby(["CourseName","CourseCategory"])["CourseRating"]
                       .mean().reset_index().sort_values("CourseRating", ascending=False).head(10))
        st.dataframe(top_courses, use_container_width=True)

    with col_r:
        st.subheader("Average Rating by Category")
        cat_rating = (df.groupby("CourseCategory")["CourseRating"]
                      .mean().reset_index().sort_values("CourseRating", ascending=False))
        fig = px.bar(cat_rating, x="CourseCategory", y="CourseRating",
                     color="CourseRating", color_continuous_scale="RdYlGn",
                     title="Average Course Rating by Category")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.subheader("Rating Distribution")
        fig2 = px.histogram(df, x="CourseRating", nbins=20,
                            color_discrete_sequence=["#636EFA"], title="Course Rating Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    with col_r2:
        st.subheader("Duration vs Rating")
        fig3 = px.scatter(df, x="CourseDuration", y="CourseRating",
                          color="CourseCategory", hover_data=["CourseName"],
                          title="Duration vs Rating")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Rating by Course Level")
    level_rating = df.groupby("CourseLevel")["CourseRating"].mean().reset_index()
    fig4 = px.bar(level_rating, x="CourseLevel", y="CourseRating",
                  color="CourseRating", color_continuous_scale="RdYlGn",
                  title="Average Rating by Course Level")
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════
# PAGE 4: LEADERBOARD
# ════════════════════════════════════════════
elif menu == "🏆 Leaderboard":
    st.title("🏆 Revenue Leaderboard")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🎓 Top Courses", "📂 Top Categories", "👨‍🏫 Top Teachers"])

    with tab1:
        revenue_courses = (df.groupby(["CourseName","CourseCategory"])["Amount"]
                           .sum().reset_index().sort_values("Amount", ascending=False).head(10))
        st.dataframe(revenue_courses, use_container_width=True)
        fig1 = px.bar(revenue_courses, x="CourseName", y="Amount",
                      color="Amount", color_continuous_scale="Viridis",
                      title="Top 10 Revenue Generating Courses")
        fig1.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        cat_revenue = (df.groupby("CourseCategory")["Amount"]
                       .sum().reset_index().sort_values("Amount", ascending=False))
        col_l, col_r = st.columns(2)
        with col_l:
            st.dataframe(cat_revenue, use_container_width=True)
        with col_r:
            fig2 = px.pie(cat_revenue, names="CourseCategory", values="Amount",
                          hole=0.5, title="Revenue Share by Category")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        teacher_revenue = (df.groupby("TeacherName")["Amount"]
                           .sum().reset_index().sort_values("Amount", ascending=False).head(10))
        st.dataframe(teacher_revenue, use_container_width=True)
        fig3 = px.bar(teacher_revenue, x="TeacherName", y="Amount",
                      color="Amount", color_continuous_scale="Turbo",
                      title="Top 10 Teachers by Revenue")
        fig3.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════
# PAGE 5: EXPLORER
# ════════════════════════════════════════════
elif menu == "🔍 Explorer":
    st.title("🔍 Data Explorer")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        search_course = st.text_input("🔎 Search Course Name")
    with col2:
        search_teacher = st.text_input("🔎 Search Teacher Name")

    explorer_df = df.copy()
    if search_course:
        explorer_df = explorer_df[explorer_df["CourseName"].str.contains(search_course, case=False, na=False)]
    if search_teacher:
        explorer_df = explorer_df[explorer_df["TeacherName"].str.contains(search_teacher, case=False, na=False)]

    st.caption(f"Showing {len(explorer_df):,} records")
    st.dataframe(explorer_df, use_container_width=True)

    csv = explorer_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Data as CSV", data=csv,
                       file_name="filtered_edupro_data.csv", mime="text/csv", use_container_width=True)


# ════════════════════════════════════════════
# PAGE 6: PREDICTIONS
# ════════════════════════════════════════════
elif menu == "🤖 Predictions":
    st.title("🤖 Course Demand & Revenue Predictor")
    st.markdown("ML models trained on EduPro historical data to predict future course performance.")
    st.divider()

    if ml is None:
        st.error("⚠️ Models nahi mile! Pehle `python ml_pipeline.py` run karo.")
    else:
        tab_pred, tab_perf = st.tabs(["🔮 Predict", "📊 Model Performance"])

        with tab_perf:
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("📈 Enrollment Models")
                enroll_df = pd.DataFrame(ml["enroll_results"]).T.reset_index().rename(columns={"index":"Model"})
                st.dataframe(enroll_df, use_container_width=True)
                fig_e = px.bar(enroll_df, x="Model", y="R2", color="R2",
                               color_continuous_scale="RdYlGn", title="R2 — Enrollment Models", text_auto=".3f")
                fig_e.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig_e, use_container_width=True)

            with col_r:
                st.subheader("💰 Revenue Models")
                rev_df = pd.DataFrame(ml["revenue_results"]).T.reset_index().rename(columns={"index":"Model"})
                st.dataframe(rev_df, use_container_width=True)
                fig_r = px.bar(rev_df, x="Model", y="R2", color="R2",
                               color_continuous_scale="RdYlGn", title="R2 — Revenue Models", text_auto=".3f")
                fig_r.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig_r, use_container_width=True)

            best_e = max(ml["enroll_results"],  key=lambda k: ml["enroll_results"][k]["R2"])
            best_r = max(ml["revenue_results"], key=lambda k: ml["revenue_results"][k]["R2"])
            c1, c2 = st.columns(2)
            c1.success(f"✅ Best Enrollment Model: **{best_e}** (R² = {ml['enroll_results'][best_e]['R2']})")
            c2.success(f"✅ Best Revenue Model: **{best_r}** (R² = {ml['revenue_results'][best_r]['R2']})")

            if ml.get("corr_info"):
                st.subheader("🔍 Correlation Analysis — Dropped Features")
                dropped = ml["corr_info"]["dropped_features"]
                final   = ml["corr_info"]["final_features"]
                st.info(f"**Removed (highly correlated >0.85):** {', '.join(dropped) if dropped else 'None'}")
                st.info(f"**Final features used ({len(final)}):** {', '.join(final)}")
                corr_df = ml["corr_info"]["correlation_matrix"]
                fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                                     title="Feature Correlation Matrix", aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)

        with tab_pred:
            st.markdown("### Enter Course Details")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Course Info**")
                course_category = st.selectbox("Category", sorted(df_raw["CourseCategory"].unique()))
                course_level    = st.selectbox("Level",    sorted(df_raw["CourseLevel"].unique()))
                course_type     = st.selectbox("Type",     sorted(df_raw["CourseType"].unique()))

            with col2:
                st.markdown("**Course Parameters**")
                course_price    = st.slider("Price ($)",        0.0, 500.0, 100.0, step=5.0)
                course_duration = st.slider("Duration (hours)", 1.0, 50.0,  20.0,  step=0.5)
                course_rating   = st.slider("Expected Rating",  1.0, 5.0,   4.0,   step=0.1)

            with col3:
                st.markdown("**Instructor Info**")
                teacher_rating  = st.slider("Instructor Rating",      1.0, 5.0, 4.0, step=0.1)
                years_exp       = st.slider("Years of Experience",    1, 25, 5)
                expertise_match = st.radio("Expertise matches Category?", ["Yes","No"], horizontal=True)

            st.divider()

            price_band      = "Free" if course_price==0 else "Low" if course_price<=100 else "Medium" if course_price<=300 else "High"
            duration_bucket = "Short" if course_duration<=10 else "Medium" if course_duration<=25 else "Long"
            rating_tier     = "Low" if course_rating<=2.5 else "Medium" if course_rating<=3.5 else "High" if course_rating<=4.2 else "Top"
            exp_bucket      = "Junior" if years_exp<=5 else "Mid" if years_exp<=12 else "Senior"
            exp_match_val   = 1 if expertise_match == "Yes" else 0

            def safe_encode(val, col):
                enc = ml["encoders"][col]
                val = str(val)
                return int(enc.transform([val])[0]) if val in enc.classes_ else 0

            all_input = {
                "CourseCategory":    safe_encode(course_category, "CourseCategory"),
                "CourseLevel":       safe_encode(course_level,    "CourseLevel"),
                "CourseType":        safe_encode(course_type,     "CourseType"),
                "CoursePrice":       course_price,
                "CourseDuration":    course_duration,
                "CourseRating":      course_rating,
                "TeacherRating":     teacher_rating,
                "YearsOfExperience": years_exp,
                "ExpertiseMatch":    exp_match_val,
                "PriceBand":         safe_encode(price_band,      "PriceBand"),
                "DurationBucket":    safe_encode(duration_bucket, "DurationBucket"),
                "RatingTier":        safe_encode(rating_tier,     "RatingTier"),
                "ExperienceBucket":  safe_encode(exp_bucket,      "ExperienceBucket"),
            }
            input_df = pd.DataFrame([all_input])[ml["features"]]

            if st.button("🚀 Predict Now", use_container_width=True, type="primary"):
                pred_enroll  = max(0, int(ml["enroll_model"].predict(input_df)[0]))
                pred_revenue = max(0.0, float(ml["revenue_model"].predict(input_df)[0]))

                st.divider()
                st.markdown("## 📈 Prediction Results")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("🎓 Predicted Enrollments", f"{pred_enroll} students")
                rc2.metric("💰 Predicted Revenue",     f"${pred_revenue:,.0f}")
                rc3.metric("💵 Revenue per Student",   f"${pred_revenue/pred_enroll:,.0f}" if pred_enroll > 0 else "N/A")

                st.divider()
                st.markdown("### 💡 Business Insights")
                insights = []
                if course_rating >= 4.2: insights.append(("✅","High course rating — strong enrollment expected."))
                else:                     insights.append(("⚠️","Improve course quality. Rating above 4.2 significantly boosts enrollments."))
                if exp_match_val == 1:    insights.append(("✅","Instructor expertise matches course category — high credibility."))
                else:                     insights.append(("⚠️","Instructor expertise mismatch may reduce learner trust."))
                if price_band in ["Free","Low"]: insights.append(("📢","Low pricing — good for enrollment volume, lower revenue per student."))
                elif price_band == "High":       insights.append(("💰","Premium pricing — fewer enrollments but higher revenue potential."))
                if years_exp >= 12:   insights.append(("🎓","Senior instructor — experience adds significant credibility."))
                elif years_exp <= 5:  insights.append(("📌","Junior instructor — consider pairing with a senior co-instructor."))
                if course_duration <= 10: insights.append(("⏱️","Short course — good for beginners; consider adding more content."))
                elif course_duration >= 35: insights.append(("📖","Long course — ensure engagement remains high throughout."))
                for icon, text in insights:
                    st.markdown(f"{icon} {text}")

                with st.expander("🔍 Engineered Feature Values Used"):
                    feat_df = pd.DataFrame({
                        "Feature": ["PriceBand","DurationBucket","RatingTier","ExperienceBucket","ExpertiseMatch"],
                        "Value":   [price_band, duration_bucket, rating_tier, exp_bucket, expertise_match]
                    })
                    st.dataframe(feat_df, use_container_width=True)


# ════════════════════════════════════════════
# PAGE 7: FEATURE IMPORTANCE  ← NEW
# ════════════════════════════════════════════
elif menu == "📈 Feature Importance":
    st.title("📈 Feature Importance Explorer")
    st.markdown("Kaun se features course demand aur revenue ko sabse zyada drive karte hain.")
    st.divider()

    if ml is None or "feat_importance" not in ml:
        st.error("⚠️ Models nahi mile! Pehle `python ml_pipeline.py` run karo.")
    else:
        fi = ml["feat_importance"]

        target_sel = st.radio("Select Target", ["Revenue", "Enrollment"], horizontal=True)
        model_sel  = st.radio("Select Model",  ["Random Forest", "Gradient Boosting"], horizontal=True)

        key = f"{'revenue' if target_sel=='Revenue' else 'enroll'}_{'rf' if model_sel=='Random Forest' else 'gb'}"
        fi_data = fi[key]

        fi_df = pd.DataFrame(list(fi_data.items()), columns=["Feature","Importance"])
        fi_df  = fi_df.sort_values("Importance", ascending=True)

        st.subheader(f"Feature Importance — {target_sel} ({model_sel})")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="RdYlGn",
                     title=f"Feature Importance for {target_sel} Prediction")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Importance Table")
        fi_df_show = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df_show["Importance %"] = (fi_df_show["Importance"] * 100).round(2).astype(str) + "%"
        st.dataframe(fi_df_show, use_container_width=True)

        st.divider()
        st.subheader("💡 Key Demand Drivers — Business Insights")

        top3 = fi_df.sort_values("Importance", ascending=False).head(3)["Feature"].tolist()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Course Price Sensitivity**")
            st.write("CoursePrice is the strongest revenue predictor. Premium courses generate significantly higher revenue per enrollment. Pricing strategy is the #1 lever for revenue optimization.")
            st.markdown("**Instructor Rating Influence**")
            st.write("TeacherRating and YearsOfExperience together explain instructor credibility impact. Higher-rated instructors consistently attract more enrollments.")
        with col2:
            st.markdown("**Course Level & Category Effects**")
            st.write("CourseCategory and CourseLevel affect both enrollment demand and revenue. AI, Business, and Project Management categories show highest revenue potential.")
            st.markdown("**Rating & Duration Impact**")
            st.write("CourseRating and CourseDuration influence enrollment decisions. Courses rated above 4.2 show significantly better enrollment numbers.")

        st.divider()
        st.subheader("🔀 Side-by-side Comparison — Revenue vs Enrollment Drivers")
        rev_rf = pd.DataFrame(list(fi["revenue_rf"].items()), columns=["Feature","Revenue Importance"])
        enr_rf = pd.DataFrame(list(fi["enroll_rf"].items()), columns=["Feature","Enrollment Importance"])
        cmp_df = rev_rf.merge(enr_rf, on="Feature").sort_values("Revenue Importance", ascending=False)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Revenue",    x=cmp_df["Feature"], y=cmp_df["Revenue Importance"],    marker_color="#636EFA"))
        fig2.add_trace(go.Bar(name="Enrollment", x=cmp_df["Feature"], y=cmp_df["Enrollment Importance"], marker_color="#EF553B"))
        fig2.update_layout(barmode="group", title="Revenue vs Enrollment Feature Importance",
                           xaxis_tickangle=-30, height=420)
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════
# PAGE 8: CATEGORY DEMAND COMPARISON  ← NEW
# ════════════════════════════════════════════
elif menu == "📂 Category Demand":
    st.title("📂 Category-Level Demand Comparison")
    st.markdown("Category wise enrollment demand, revenue, aur avg rating ka comparison.")
    st.divider()

    cat_summary = df.groupby("CourseCategory").agg(
        Enrollments  = ("TransactionID", "count"),
        TotalRevenue = ("Amount",        "sum"),
        AvgRevenue   = ("Amount",        "mean"),
        AvgRating    = ("CourseRating",  "mean"),
        Courses      = ("CourseID",      "nunique"),
    ).reset_index().sort_values("TotalRevenue", ascending=False)

    cat_summary["RevenuePerEnrollment"] = (cat_summary["TotalRevenue"] / cat_summary["Enrollments"]).round(2)

    # KPI cards
    top_cat     = cat_summary.iloc[0]["CourseCategory"]
    top_demand  = cat_summary.sort_values("Enrollments", ascending=False).iloc[0]["CourseCategory"]
    top_rating  = cat_summary.sort_values("AvgRating",   ascending=False).iloc[0]["CourseCategory"]

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Highest Revenue Category",    top_cat)
    c2.metric("🎓 Highest Demand Category",     top_demand)
    c3.metric("⭐ Highest Rated Category",       top_rating)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Enrollments","💰 Revenue","⭐ Rating","📋 Full Table"])

    with tab1:
        st.subheader("Enrollment Demand by Category")
        enroll_sorted = cat_summary.sort_values("Enrollments", ascending=False)
        fig1 = px.bar(enroll_sorted, x="CourseCategory", y="Enrollments",
                      color="Enrollments", color_continuous_scale="Blues",
                      title="Enrollment Count by Category", text_auto=True)
        fig1.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig1, use_container_width=True)

        fig1b = px.pie(enroll_sorted, names="CourseCategory", values="Enrollments",
                       hole=0.4, title="Enrollment Share by Category")
        st.plotly_chart(fig1b, use_container_width=True)

    with tab2:
        st.subheader("Revenue Comparison by Category")
        col_l, col_r = st.columns(2)
        with col_l:
            fig2 = px.bar(cat_summary, x="CourseCategory", y="TotalRevenue",
                          color="TotalRevenue", color_continuous_scale="Viridis",
                          title="Total Revenue by Category")
            fig2.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)
        with col_r:
            fig2b = px.bar(cat_summary, x="CourseCategory", y="RevenuePerEnrollment",
                           color="RevenuePerEnrollment", color_continuous_scale="RdYlGn",
                           title="Revenue per Enrollment (Efficiency)")
            fig2b.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig2b, use_container_width=True)

    with tab3:
        st.subheader("Average Course Rating by Category")
        fig3 = px.bar(cat_summary.sort_values("AvgRating", ascending=False),
                      x="CourseCategory", y="AvgRating",
                      color="AvgRating", color_continuous_scale="RdYlGn",
                      title="Average Rating by Category", text_auto=".2f")
        fig3.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("Complete Category Summary Table")
        st.dataframe(cat_summary.reset_index(drop=True), use_container_width=True)
        csv = cat_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Category Summary", data=csv,
                           file_name="category_demand_summary.csv", mime="text/csv")
