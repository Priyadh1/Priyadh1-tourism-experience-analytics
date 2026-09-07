import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
 
st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")
 
# ============================================================
# LOAD EVERYTHING (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_assets():
    clf_model = joblib.load("best_classification_model.pkl")
    reg_model = joblib.load("best_regression_model.pkl")
    encoders = joblib.load("feature_encoders.pkl")
    xgb_le = joblib.load("xgb_target_label_encoder.pkl")
    scaler_reg = joblib.load("scaler_regression.pkl")
    scaler_clf = joblib.load("scaler_classification.pkl")
    feature_cols_reg = joblib.load("feature_cols_reg.pkl")
    feature_cols_clf = joblib.load("feature_cols_clf.pkl")
    similarity_df = joblib.load("attraction_similarity.pkl")
    return clf_model, reg_model, encoders, xgb_le, scaler_reg, scaler_clf, feature_cols_reg, feature_cols_clf, similarity_df
 
@st.cache_data
def load_lookups():
    continent = pd.read_excel("Continent.xlsx")
    region = pd.read_excel("Region.xlsx")
    country = pd.read_excel("Country.xlsx")
    city = pd.read_excel("City.xlsx")
    item = pd.read_excel("Item.xlsx")
    type_df = pd.read_excel("Type.xlsx")
    mode = pd.read_excel("Mode.xlsx")
    attraction_stats = pd.read_csv("attraction_stats.csv")
    user_stats = pd.read_csv("user_stats.csv")
 
    for d, col in [(continent, "Continent"), (country, "Country"), (region, "Region"), (mode, "VisitMode")]:
        d[col] = d[col].replace("-", "Unspecified")
    city["CityName"] = city["CityName"].fillna("Unknown").replace("-", "Unspecified")
 
    item_full = item.merge(type_df, on="AttractionTypeId", how="left")
    item_full = item_full.merge(attraction_stats, on="AttractionId", how="left")
    item_full["AttractionPopularity"] = item_full["AttractionPopularity"].fillna(0)
    item_full["AttractionAvgRating"] = item_full["AttractionAvgRating"].fillna(item_full["AttractionAvgRating"].mean())
 
    return continent, region, country, city, item_full, type_df, mode, user_stats
 
clf_model, reg_model, encoders, xgb_le, scaler_reg, scaler_clf, feature_cols_reg, feature_cols_clf, similarity_df = load_assets()
continent, region, country, city, item_full, type_df, mode, user_stats = load_lookups()
 
DEFAULT_USER_VISIT_COUNT = user_stats["UserVisitCount"].median()
 
# ============================================================
# SIDEBAR NAV
# ============================================================
st.sidebar.title("🌍 Tourism Experience Analytics")
page = st.sidebar.radio("Navigate", ["✈️ Plan My Trip", "📊 Explore Insights", "🧭 Discover Attractions", "🤖 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.caption("Built by Priyadharshini Murugan")
 
# ============================================================
# PAGE 1 — PLAN MY TRIP (Regression + Classification)
# ============================================================
if page == "✈️ Plan My Trip":
    st.title("✈️ Plan My Trip")
    st.write("Tell us a bit about your trip, and we'll predict how you'll rate it and how you're likely traveling.")
 
    col1, col2 = st.columns(2)
    with col1:
        sel_continent = st.selectbox("Continent", sorted(continent["Continent"].unique()))
        countries_in_continent = country.merge(region, on="RegionId")
        countries_in_continent = countries_in_continent[countries_in_continent["Continent"].isin(
            continent[continent["Continent"] == sel_continent]["ContinentId"].apply(
                lambda cid: region[region["ContinentId"] == cid]["Region"].tolist()
            ).sum()
        )] if False else countries_in_continent  # placeholder guard
 
        cont_id = continent[continent["Continent"] == sel_continent]["ContinentId"].iloc[0]
        regions_avail = region[region["ContinentId"] == cont_id]
        sel_region = st.selectbox("Region", sorted(regions_avail["Region"].unique()) if len(regions_avail) else ["Unspecified"])
 
        region_id = regions_avail[regions_avail["Region"] == sel_region]["RegionId"].iloc[0] if len(regions_avail) else 0
        countries_avail = country[country["RegionId"] == region_id]
        sel_country = st.selectbox("Country", sorted(countries_avail["Country"].unique()) if len(countries_avail) else ["Unspecified"])
 
    with col2:
        country_id = countries_avail[countries_avail["Country"] == sel_country]["CountryId"].iloc[0] if len(countries_avail) else 0
        cities_avail = city[city["CountryId"] == country_id]
        sel_city = st.selectbox("City", sorted(cities_avail["CityName"].unique()) if len(cities_avail) else ["Unspecified"])
 
        sel_type = st.selectbox("Attraction Type", sorted(type_df["AttractionType"].unique()))
        from datetime import datetime
        current_year = datetime.now().year
        sel_year = st.slider("Visit Year", 2013, current_year + 5, current_year)
        sel_month = st.selectbox("Visit Month", list(range(1, 13)), format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
 
    st.markdown("---")
 
    if st.button("🔮 Predict My Trip", type="primary", width='stretch'):
        def safe_encode(col, value):
            le = encoders[col]
            if value in le.classes_:
                return le.transform([value])[0]
            return le.transform([le.classes_[0]])[0]
 
        type_attractions = item_full[item_full["AttractionType"] == sel_type]
        avg_popularity = type_attractions["AttractionPopularity"].mean() if len(type_attractions) else 0
        avg_att_rating = type_attractions["AttractionAvgRating"].mean() if len(type_attractions) else item_full["AttractionAvgRating"].mean()
 
        base_row = {
            "VisitYear": sel_year,
            "VisitMonth": sel_month,
            "Continent_enc": safe_encode("Continent", sel_continent),
            "Region_enc": safe_encode("Region", sel_region),
            "Country_enc": safe_encode("Country", sel_country),
            "CityName_enc": safe_encode("CityName", sel_city),
            "AttractionType_enc": safe_encode("AttractionType", sel_type),
            "UserVisitCount": DEFAULT_USER_VISIT_COUNT,
            "AttractionPopularity": avg_popularity,
        }
 
        # --- Regression ---
        X_reg_input = pd.DataFrame([base_row])[feature_cols_reg]
        scale_cols_reg = ["VisitYear", "VisitMonth", "UserVisitCount", "AttractionPopularity"]
        X_reg_scaled = X_reg_input.copy()
        X_reg_scaled[scale_cols_reg] = scaler_reg.transform(X_reg_input[scale_cols_reg])
        pred_rating = reg_model.predict(X_reg_input)[0]  # tree model uses unscaled
        pred_rating = float(np.clip(pred_rating, 1, 5))
 
        # --- Classification ---
        clf_row = dict(base_row)
        clf_row["AttractionAvgRating"] = avg_att_rating
        clf_row["UserAvgRating"] = user_stats["UserAvgRating"].mean()
        X_clf_input = pd.DataFrame([clf_row])[feature_cols_clf]
        probs = clf_model.predict_proba(X_clf_input)[0]
        classes_decoded = xgb_le.inverse_transform(clf_model.classes_)
        mode_names = mode.set_index("VisitModeId")["VisitMode"].to_dict()
        prob_df = pd.DataFrame({
            "VisitMode": [mode_names.get(c, str(c)) for c in classes_decoded],
            "Probability": probs
        }).sort_values("Probability", ascending=False)
 
        st.success("Here's what we predict for your trip!")
        r1, r2 = st.columns([1, 1.4])
        with r1:
            st.metric("⭐ Predicted Rating", f"{pred_rating:.1f} / 5")
            st.progress(pred_rating / 5)
            top_mode = prob_df.iloc[0]
            st.metric("🧳 Most Likely Visit Mode", top_mode["VisitMode"], f"{top_mode['Probability']*100:.0f}% confidence")
        with r2:
            fig = px.bar(prob_df, x="Probability", y="VisitMode", orientation="h",
                         color="Probability", color_continuous_scale="YlGn", title="Visit Mode Likelihood")
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
            st.plotly_chart(fig, width='stretch')
 
        st.markdown("### 🏝️ Recommended attractions for you")
        top_attractions = type_attractions.sort_values("AttractionAvgRating", ascending=False).head(4)
        cols = st.columns(min(4, max(1, len(top_attractions))))
        for i, (_, row) in enumerate(top_attractions.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"**{row['Attraction']}**")
                st.caption(row["AttractionType"])
                st.write(f"⭐ {row['AttractionAvgRating']:.1f} avg rating")
 
# ============================================================
# PAGE 2 — EXPLORE INSIGHTS (interactive EDA)
# ============================================================
elif page == "📊 Explore Insights":
    st.title("📊 Explore Insights")
    st.write("Filter and explore the tourism dataset interactively.")
 
    f1, f2 = st.columns(2)
    with f1:
        filter_continent = st.multiselect("Filter by Continent", sorted(continent["Continent"].unique()))
    with f2:
        filter_type = st.multiselect("Filter by Attraction Type", sorted(type_df["AttractionType"].unique()))
 
    plot_df = item_full.copy()
    if filter_type:
        plot_df = plot_df[plot_df["AttractionType"].isin(filter_type)]
 
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(plot_df.sort_values("AttractionPopularity", ascending=False).head(10),
                      x="AttractionPopularity", y="Attraction", orientation="h",
                      color="AttractionPopularity", color_continuous_scale="YlGn",
                      title="Top 10 Most Visited Attractions")
        st.plotly_chart(fig1, width='stretch')
    with c2:
        fig2 = px.bar(plot_df.groupby("AttractionType")["AttractionAvgRating"].mean().sort_values().reset_index(),
                      x="AttractionAvgRating", y="AttractionType", orientation="h",
                      color="AttractionAvgRating", color_continuous_scale="YlGn",
                      title="Average Rating by Attraction Type")
        st.plotly_chart(fig2, width='stretch')
 
    st.markdown("### Visit Mode Breakdown")
    mode_counts = mode[mode["VisitMode"] != "Unspecified"]
    fig3 = px.pie(mode_counts, names="VisitMode", title="Visit Mode Categories", hole=0.4)
    st.plotly_chart(fig3, width='stretch')
 
# ============================================================
# PAGE 3 — DISCOVER ATTRACTIONS (Recommender)
# ============================================================
elif page == "🧭 Discover Attractions":
    st.title("🧭 Discover Attractions")
    st.write("Pick an attraction you like — we'll recommend similar ones based on how other travelers rated them.")
 
    attraction_names = item_full.sort_values("Attraction")["Attraction"].tolist()
    sel_attraction = st.selectbox("I like...", attraction_names)
 
    sel_id = item_full[item_full["Attraction"] == sel_attraction]["AttractionId"].iloc[0]
 
    if sel_id in similarity_df.columns:
        scores = similarity_df[sel_id].drop(sel_id, errors="ignore").sort_values(ascending=False).head(5)
        rec_df = item_full[item_full["AttractionId"].isin(scores.index)].copy()
        rec_df["Similarity"] = rec_df["AttractionId"].map(scores)
        rec_df = rec_df.sort_values("Similarity", ascending=False)
 
        st.markdown(f"### Because you like **{sel_attraction}**")
        cols = st.columns(min(5, max(1, len(rec_df))))
        for i, (_, row) in enumerate(rec_df.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"**{row['Attraction']}**")
                st.caption(row["AttractionType"])
                st.write(f"🔗 {row['Similarity']*100:.0f}% similar")
                st.write(f"⭐ {row['AttractionAvgRating']:.1f} avg rating")
    else:
        st.info("Not enough data to recommend similar attractions for this one yet.")
 
# ============================================================
# PAGE 4 — MODEL PERFORMANCE (transparency)
# ============================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.write("A transparent look at how the models were built and how well they perform.")
 
    st.markdown("### Classification — Predicting Visit Mode")
    clf_hist = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Random Forest (Tuned)", "XGBoost (Tuned)"],
        "Test Accuracy": [0.2814, 0.3493, 0.3754, 0.4000, 0.4255]
    })
    fig4 = px.bar(clf_hist, x="Model", y="Test Accuracy", color="Test Accuracy",
                  color_continuous_scale="YlGn", title="Classification Accuracy Across Models")
    fig4.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig4, width='stretch')
    st.caption("Final model: XGBoost (Tuned) — 42.6% test accuracy, more than 2x random-guessing baseline (20% for 5 classes).")
 
    st.markdown("### Regression — Predicting Rating")
    reg_hist = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest (Tuned)", "XGBoost (Tuned)"],
        "Test R2": [0.0404, 0.1278, 0.1268]
    })
    fig5 = px.bar(reg_hist, x="Model", y="Test R2", color="Test R2",
                  color_continuous_scale="YlGn", title="Regression R² Across Models")
    fig5.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig5, width='stretch')
    st.caption("Final model: Random Forest (Tuned) — R² 12.8%, RMSE 0.91.")
 
    st.markdown("### Honest Note")
    st.info(
        "Visit Mode and Rating are shaped by personal travel choices, not just location and timing. "
        "This model set genuinely improved through feature engineering (user/attraction behavior stats) "
        "and hyperparameter tuning, and results are reported transparently rather than overstated."
    )
 
