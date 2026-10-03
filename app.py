# ============================================================
# STREAMLIT APP — Household Sanitation Service Status Predictor
# Best model: XGBoost pipeline | Youden threshold = 0.5533
# ============================================================
import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Household Sanitation Service Status Predictor",
    page_icon="🚽",
    layout="wide",
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_pipeline():
    path = "best_model_pipeline.pkl"
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found in the app directory.")
        st.stop()
    return joblib.load(path)

pipeline = load_pipeline()

# ---------- EXACT MODEL INPUT COLUMNS (34) ----------
COLUMN_ORDER = [
    'household_wealth_index',
    'highest_educational_attainment',
    'place_of_residence',
    'source_of_drinking_water',
    'time_to_get_water_(in_minutes)',
    'has_electricity',
    'sex_of_head_of_household',
    'presence_of_water_at_hand_washing_place',
    'household_water_treatment',
    'share_toilet_with_other',
    'observed_handwashing_facility',
    'household_media_exposure',
    'number_of_households_Members',
    'cooking_fuel_type',
    'housing_material_status',
    'community_level_education',
    'community_level_media_exposure',
    'community_level_poverty',
    'amhara_region',
    'benishangul-gumuz_region',
    'central_ethiopia_region',
    'dire_dawa_region',
    'harari_region',
    'oromia_region',
    'sidama_region',
    'south_ethiopia_region',
    'south_west_ethiopia_region',
    'tigray_region',
    'current_marital_status_married',
    'location_of_source_for_water_in_own_yard/plot',
    'children_under5_group_1-2_children',
    'children_under5_group_=3_children',
    'hh_head_age_group_35-60_years',
    'hh_head_age_group_60_years',
]

# ---------- YOUDEN THRESHOLD ----------
YOUDEN_THRESHOLD = 0.5533

# ---------- REGION MAP (reference = Addis Ababa) ----------
REGION_DUMMIES = {
    "Addis Ababa":         None,
    "Amhara":              "amhara_region",
    "Benishangul-Gumuz":   "benishangul-gumuz_region",
    "Central Ethiopia":    "central_ethiopia_region",
    "Dire Dawa":           "dire_dawa_region",
    "Harari":              "harari_region",
    "Oromia":              "oromia_region",
    "Sidama":              "sidama_region",
    "South Ethiopia":      "south_ethiopia_region",
    "South West Ethiopia": "south_west_ethiopia_region",
    "Tigray":              "tigray_region",
}

# ---------- BUILD FEATURE ROW ----------
def build_features(
    wealth, education, residence, water_source, water_time,
    electricity, sex, hw_water, water_treat, share_toilet,
    hw_obs, hh_media, hh_size, fuel_type, housing_mat,
    comm_edu, comm_media, comm_poverty, region,
    marital_status, water_location, children_group, age_group,
):
    f = {c: 0 for c in COLUMN_ORDER}

    wealth_map = {"Poorest": 0, "Poorer": 1, "Middle": 2, "Richer": 3, "Richest": 4}
    edu_map    = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
    f['household_wealth_index']         = wealth_map[wealth]
    f['highest_educational_attainment'] = edu_map[education]

    # Binary-encoded HH size and water time (matches the coding scheme)
    f['time_to_get_water_(in_minutes)'] = 1 if water_time == "≤ 30 Minutes" else 0
    f['number_of_households_Members']   = 1 if hh_size == "≥ 4 members" else 0

    f['place_of_residence']                      = 1 if residence == "Urban" else 0
    f['source_of_drinking_water']                = 1 if water_source == "Improved" else 0
    f['has_electricity']                         = 1 if electricity == "Yes" else 0
    f['sex_of_head_of_household']                = 1 if sex == "Male" else 0
    f['presence_of_water_at_hand_washing_place'] = 1 if hw_water == "Available" else 0
    f['household_water_treatment']               = 1 if water_treat == "Yes" else 0
    f['share_toilet_with_other']                 = 1 if share_toilet == "Yes" else 0
    f['observed_handwashing_facility']           = 1 if hw_obs == "Yes" else 0
    f['household_media_exposure']                = 1 if hh_media == "Yes" else 0
    f['cooking_fuel_type']                       = 1 if fuel_type == "Clean fuel" else 0
    f['housing_material_status']                 = 1 if housing_mat == "Improved" else 0
    f['community_level_education']               = 1 if comm_edu == "High" else 0
    f['community_level_media_exposure']          = 1 if comm_media == "High" else 0
    f['community_level_poverty']                 = 1 if comm_poverty == "High" else 0

    dummy = REGION_DUMMIES.get(region)
    if dummy is not None:
        f[dummy] = 1

    f['current_marital_status_married'] = 1 if marital_status == "Married" else 0
    f['location_of_source_for_water_in_own_yard/plot'] = (
        1 if water_location == "In own yard/plot" else 0
    )

    if children_group == "1-2 children":
        f['children_under5_group_1-2_children'] = 1
    elif children_group == ">=3 children":
        f['children_under5_group_=3_children'] = 1

    if age_group == "35-60 years":
        f['hh_head_age_group_35-60_years'] = 1
    elif age_group == ">60 years":
        f['hh_head_age_group_60_years'] = 1

    # ⬇ THE KEY FIX: return a DataFrame, NOT a numpy array
    return pd.DataFrame([f], columns=COLUMN_ORDER)


# ---------- UI ----------
st.title("🚽 Household Sanitation Service Status Predictor")
st.markdown(
    "This tool uses the **best-performing XGBoost model** to estimate the probability "
    "that a household has **unimproved** sanitation service status. "
    f"Classifications use the optimal **Youden threshold = {YOUDEN_THRESHOLD}**."
)

with st.sidebar:
    st.header("Enter household information")

    st.subheader("Household demographics")
    sex            = st.radio("Sex of head of household", ["Female", "Male"])
    hh_size        = st.radio("Number of household members", ["< 4 members", "≥ 4 members"])
    age_group      = st.selectbox("Age group of head of household",
                                  ["<35 years", "35-60 years", ">60 years"])
    marital_status = st.selectbox("Current marital status",
                                  ["Never married", "Married", "Divorced", "Widowed"])
    education      = st.selectbox("Highest educational attainment",
                                  ["No education", "Primary", "Secondary", "Higher"])
    children_group = st.selectbox("Children under 5 in household",
                                  ["No child", "1-2 children", ">=3 children"])

    st.subheader("Water & sanitation")
    water_source   = st.radio("Source of drinking water", ["Unimproved", "Improved"])
    water_location = st.radio("Location of source for water",
                              ["Elsewhere", "In own yard/plot"])
    water_time     = st.radio("Time to get water (minutes)",
                              ["> 30 Minutes", "≤ 30 Minutes"])
    water_treat    = st.radio("Household water treatment", ["No", "Yes"])
    share_toilet   = st.radio("Share toilet with other households", ["No", "Yes"])
    hw_water       = st.radio("Presence of water at hand washing place",
                              ["Not available", "Available"])
    hw_obs         = st.radio("Observed handwashing facility", ["No", "Yes"])

    st.subheader("Socioeconomic")
    wealth       = st.selectbox("Household wealth index",
                                ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
    electricity  = st.radio("Has electricity", ["No", "Yes"])
    fuel_type    = st.radio("Cooking fuel type", ["Solid fuel", "Clean fuel"])
    housing_mat  = st.radio("Housing material status", ["Unimproved", "Improved"])
    hh_media     = st.radio("Household media exposure", ["No", "Yes"])

    st.subheader("Community level")
    residence      = st.radio("Place of residence", ["Rural", "Urban"])
    comm_media     = st.selectbox("Community-level media exposure", ["Low", "High"])
    comm_poverty   = st.selectbox("Community-level poverty", ["Low", "High"])
    comm_edu       = st.selectbox("Community-level education", ["Low", "High"])
    region         = st.selectbox("Region", list(REGION_DUMMIES.keys()))

    predict_btn = st.button("Predict Sanitation Service Status", type="primary")


# ---------- PREDICT ----------
if predict_btn:
    X_input = build_features(
        wealth, education, residence, water_source, water_time,
        electricity, sex, hw_water, water_treat, share_toilet,
        hw_obs, hh_media, hh_size, fuel_type, housing_mat,
        comm_edu, comm_media, comm_poverty, region,
        marital_status, water_location, children_group, age_group,
    )

    try:
        prob_unimproved = float(pipeline.predict_proba(X_input)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred  = 1 if prob_unimproved >= YOUDEN_THRESHOLD else 0
    label = "Unimproved sanitation" if pred == 1 else "Improved sanitation"

    st.subheader("Prediction result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", label)
    c2.metric("P(unimproved)", f"{prob_unimproved:.2%}")
    c3.metric("Youden threshold", f"{YOUDEN_THRESHOLD:.4f}")

    if pred == 1:
        st.error(
            f"**{label}** — probability of unimproved sanitation "
            f"({prob_unimproved:.1%}) ≥ threshold ({YOUDEN_THRESHOLD})."
        )
    else:
        st.success(
            f"**{label}** — probability of unimproved sanitation "
            f"({prob_unimproved:.1%}) < threshold ({YOUDEN_THRESHOLD})."
        )

    with st.expander("Show feature vector (developers only)"):
        st.dataframe(X_input)

else:
    st.info("Fill in the sidebar and click **Predict Sanitation Service Status**.")

st.markdown("---")
st.caption(
    "Model: XGBoost | Trained on DHS data from Ethiopia | "
    f"Youden threshold = {YOUDEN_THRESHOLD}"
)
