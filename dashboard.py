import json
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ========= CONFIG =========
DATA_PATH = "data/suffolk_zip_acs_2023.csv"
GEOJSON_PATH = "geo/suffolk_zctas.geojson"  
GEOJSON_ZIP_PROPERTY = "GEOID"              # field in GeoJSON with ZIP/ZCTA code


# ========= HELPER FUNCTIONS =========
def fmt_currency(x):
    if pd.isna(x):
        return "–"
    return f"${x:,.0f}"


def fmt_int(x):
    if pd.isna(x):
        return "–"
    return f"{x:,.0f}"


def fmt_percent(x, decimals=1):
    if pd.isna(x):
        return "–"
    return f"{x:.{decimals}f}%"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"zip": str})
    df["zip"] = df["zip"].str.zfill(5)
    return df


@st.cache_data
def cached_load_data(path: str) -> pd.DataFrame:
    return load_data(path)


def generate_pdf(report_text: str) -> BytesIO:
    """
    Create a simple multi-paragraph PDF from text and return as BytesIO.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    textobject = c.beginText(40, height - 40)
    textobject.setFont("Helvetica", 11)

    max_width = 90  # approximate characters per line

    for paragraph in report_text.split("\n"):
        if not paragraph.strip():
            textobject.textLine("")
            continue
        line = paragraph
        while len(line) > max_width:
            textobject.textLine(line[:max_width])
            line = line[max_width:]
        textobject.textLine(line)
        textobject.textLine("")

    c.drawText(textobject)
    c.save()
    buffer.seek(0)
    return buffer


# ========= APP SETUP =========
st.set_page_config(page_title="Suffolk County ZIP Dashboard", layout="wide")

df = cached_load_data(DATA_PATH)

st.title("Suffolk County, NY – ZIP Code Desirability Dashboard For Computations and Visualizations Project 2")

st.markdown(
    """
By – Bhumika Channakeshava


This dashboard summarizes **ZIP Code (ZCTA) conditions in Suffolk County, NY** using a composite
desirability score based on **income, education, poverty, unemployment, housing costs, population, and housing cost burden**.

All data are from the **U.S. Census Bureau, 2019–2023 American Community Survey (ACS) 5-year estimates** via the Census API.
"""
)


# ========= SIDEBAR FILTERS =========
st.sidebar.header("Filters")

zip_prefixes = sorted(df["zip"].str[:3].unique())
selected_prefixes = st.sidebar.multiselect(
    "ZIP code prefixes",
    options=zip_prefixes,
    default=zip_prefixes,
)

income_min = float(df["income_median"].min(skipna=True))
income_max = float(df["income_median"].max(skipna=True))
income_range = st.sidebar.slider(
    "Median household income range ($)",
    min_value=int(income_min),
    max_value=int(income_max),
    value=(int(income_min), int(income_max)),
    step=1000,
)

pop_min = float(df["population_total"].min(skipna=True))
pop_max = float(df["population_total"].max(skipna=True))
pop_range = st.sidebar.slider(
    "Population range",
    min_value=int(pop_min),
    max_value=int(pop_max),
    value=(int(pop_min), int(pop_max)),
    step=100,
)

filtered_df = df[
    (df["zip"].str[:3].isin(selected_prefixes))
    & (df["income_median"].between(income_range[0], income_range[1]))
    & (df["population_total"].between(pop_range[0], pop_range[1]))
].copy()

st.caption(
    f"Showing **{len(filtered_df)}** ZIP codes after applying filters "
    f"(out of {len(df)} total Suffolk-style ZCTAs)."
)

# ========= KEY INSIGHTS & FINDINGS =========
st.subheader("Key Insights & Findings")

if filtered_df.empty:
    st.info("No ZIP codes match the current filters. Adjust the filters in the sidebar to see insights.")
    key_insights_md = "No data available under current filters."
else:
    strongest = filtered_df.sort_values("composite_score", ascending=False).iloc[0]
    weakest = filtered_df.sort_values("composite_score", ascending=True).iloc[0]

    income_max_row = filtered_df.loc[filtered_df["income_median"].idxmax()]
    income_min_row = filtered_df.loc[filtered_df["income_median"].idxmin()]
    income_gap = income_max_row["income_median"] - income_min_row["income_median"]

    aff_row = filtered_df.loc[filtered_df["housing_cost_burden_pct"].idxmax()]

    corr_text = "Not enough data to compute a meaningful correlation."
    if (
        filtered_df["income_median"].notna().sum() > 3
        and filtered_df["poverty_pct"].notna().sum() > 3
    ):
        corr_val = filtered_df[["income_median", "poverty_pct"]].corr().loc["income_median", "poverty_pct"]
        if np.isnan(corr_val):
            corr_text = "Correlation could not be computed due to missing values."
        else:
            abs_corr = abs(corr_val)
            if abs_corr >= 0.7:
                strength = "strong"
            elif abs_corr >= 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            direction = "negative" if corr_val < 0 else "positive"
            corr_text = (
                f"The correlation between median income and poverty rate among the selected ZIP codes "
                f"is **{corr_val:.2f}**, indicating a **{strength} {direction} relationship**. "
            )
            if corr_val < 0:
                corr_text += "As income increases, poverty tends to decrease, which is consistent with expectations."
            else:
                corr_text += (
                    "Higher income is associated with higher poverty in this subset, suggesting heterogeneity or outliers."
                )

    key_insights_md = f"""
- The highest composite score in the current view is **{strongest['composite_score']:.1f}** in ZIP **{strongest['zip']}**, combining relatively favorable values for income, education, and housing indicators.

- The lowest composite score is **{weakest['composite_score']:.1f}** in ZIP **{weakest['zip']}**, reflecting comparatively weaker performance across multiple variables.

- Median household income ranges from `${income_min_row['income_median']:,.0f}` in ZIP **{income_min_row['zip']}** to `${income_max_row['income_median']:,.0f}` in ZIP **{income_max_row['zip']}**, a gap of roughly `${income_gap:,.0f}` between the lowest- and highest-income ZIP codes.

- The ZIP with the greatest renter housing cost burden is **{aff_row['zip']}**, where approximately **{aff_row['housing_cost_burden_pct']:.1f}%** of renter households spend ≥30% of their income on rent.

- {corr_text}
"""

    st.markdown(key_insights_md)

# ========= TOP ZIP SUMMARY =========
st.subheader("Top ZIP codes by composite score")

top_n = st.slider("Show top N ZIP codes:", min_value=5, max_value=30, value=10, step=5)
top_df = filtered_df.sort_values("composite_score", ascending=False).head(top_n).copy()

top_display = top_df[
    [
        "zip",
        "composite_score",
        "income_median",
        "edu_bachelor_plus_pct",
        "poverty_pct",
        "unemployment_pct",
        "median_gross_rent",
        "median_home_value",
        "population_total",
        "housing_cost_burden_pct",
    ]
].copy()

top_display.rename(
    columns={
        "zip": "ZIP",
        "composite_score": "Score",
        "income_median": "Median income",
        "edu_bachelor_plus_pct": "% bachelor+",
        "poverty_pct": "Poverty rate",
        "unemployment_pct": "Unemployment rate",
        "median_gross_rent": "Median rent",
        "median_home_value": "Median home value",
        "population_total": "Population",
        "housing_cost_burden_pct": "Renters ≥30% income on rent",
    },
    inplace=True,
)

top_display["Median income"] = top_display["Median income"].apply(fmt_currency)
top_display["Median rent"] = top_display["Median rent"].apply(fmt_currency)
top_display["Median home value"] = top_display["Median home value"].apply(fmt_currency)
top_display["Population"] = top_display["Population"].apply(fmt_int)
top_display["% bachelor+"] = top_display["% bachelor+"].apply(fmt_percent)
top_display["Poverty rate"] = top_display["Poverty rate"].apply(fmt_percent)
top_display["Unemployment rate"] = top_display["Unemployment rate"].apply(fmt_percent)
top_display["Renters ≥30% income on rent"] = top_display[
    "Renters ≥30% income on rent"
].apply(fmt_percent)

st.dataframe(top_display, use_container_width=True)

# ========= MAP: ZIP COMPOSITE SCORE =========
st.subheader("Map: ZIP code composite score")

if GEOJSON_PATH is not None:
    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        feature_id_key = f"properties.{GEOJSON_ZIP_PROPERTY}"

        map_fig = px.choropleth(
            filtered_df,
            geojson=geojson,
            locations="zip",
            featureidkey=feature_id_key,
            color="composite_score",
            hover_data={
                "zip": True,
                "composite_score": ":.1f",
                "income_median": ":,.0f",
                "edu_bachelor_plus_pct": ":.1f",
                "poverty_pct": ":.1f",
                "unemployment_pct": ":.1f",
                "median_gross_rent": ":,.0f",
                "median_home_value": ":,.0f",
                "population_total": ":,.0f",
                "housing_cost_burden_pct": ":.1f",
            },
            color_continuous_scale="Viridis",
            title="ZIP code composite scores (higher = more desirable)",
        )
        map_fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(map_fig, use_container_width=True)
    except Exception as e:
        st.warning("GeoJSON not found or invalid. Showing charts without the map.")
        st.caption(f"Details: {e}")
else:
    st.info(
        "No GeoJSON configured. Add Suffolk County ZCTA GeoJSON and update GEOJSON_PATH "
        "to enable the choropleth map."
    )

# ========= VARIABLE-SPECIFIC BAR CHART =========
st.subheader("Compare variables across ZIP codes")

variable_options = {
    "income_median": "Median household income",
    "edu_bachelor_plus_pct": "% adults 25+ with bachelor's+",
    "poverty_pct": "Poverty rate (%)",
    "unemployment_pct": "Unemployment rate (%)",
    "median_gross_rent": "Median gross rent",
    "median_home_value": "Median home value",
    "population_total": "Population (proxy for density)",
    "housing_cost_burden_pct": "Renter housing cost burden (%)",
}

var_key = st.selectbox(
    "Choose a variable to visualize:",
    options=list(variable_options.keys()),
    format_func=lambda k: variable_options[k],
)

sorted_df = filtered_df.sort_values(var_key, ascending=False)

bar_fig = px.bar(
    sorted_df,
    x="zip",
    y=var_key,
    hover_data={
        "zip": True,
        "composite_score": ":.1f",
        "income_median": ":,.0f",
        "edu_bachelor_plus_pct": ":.1f",
        "poverty_pct": ":.1f",
        "unemployment_pct": ":.1f",
        "median_gross_rent": ":,.0f",
        "median_home_value": ":,.0f",
        "population_total": ":,.0f",
        "housing_cost_burden_pct": ":.1f",
    },
    labels={"zip": "ZIP code", var_key: variable_options[var_key]},
    title=f"{variable_options[var_key]} by ZIP code",
)
bar_fig.update_layout(xaxis={"type": "category"})
st.plotly_chart(bar_fig, use_container_width=True)

# ========= SCATTER: INCOME vs POVERTY =========
st.subheader("Income vs. Poverty rate")

scatter_fig = px.scatter(
    filtered_df,
    x="income_median",
    y="poverty_pct",
    size="population_total",
    color="composite_score",
    hover_data={
        "zip": True,
        "income_median": ":,.0f",
        "poverty_pct": ":.1f",
        "population_total": ":,.0f",
        "composite_score": ":.1f",
    },
    labels={
        "income_median": "Median household income ($)",
        "poverty_pct": "Poverty rate (%)",
        "composite_score": "Composite score",
    },
    title="Income vs Poverty (bubble size = population)",
)
st.plotly_chart(scatter_fig, use_container_width=True)

# ========= FULL SCORE BREAKDOWN TABLE =========
st.subheader("Full score breakdown by ZIP")

score_cols = [
    "zip",
    "composite_score",
    "income_median",
    "income_scaled",
    "edu_bachelor_plus_pct",
    "edu_scaled",
    "poverty_pct",
    "poverty_scaled",
    "unemployment_pct",
    "unemployment_scaled",
    "median_gross_rent",
    "rent_scaled",
    "median_home_value",
    "home_value_scaled",
    "population_total",
    "population_scaled",
    "housing_cost_burden_pct",
    "housing_cost_burden_scaled",
]

score_df = filtered_df[score_cols].copy()
score_df.rename(
    columns={
        "zip": "ZIP",
        "composite_score": "Score",
        "income_median": "Median income",
        "income_scaled": "Income (0–100)",
        "edu_bachelor_plus_pct": "% bachelor+",
        "edu_scaled": "Education (0–100)",
        "poverty_pct": "Poverty rate",
        "poverty_scaled": "Poverty (0–100)",
        "unemployment_pct": "Unemployment rate",
        "unemployment_scaled": "Unemployment (0–100)",
        "median_gross_rent": "Median rent",
        "rent_scaled": "Rent (0–100)",
        "median_home_value": "Median home value",
        "home_value_scaled": "Home value (0–100)",
        "population_total": "Population",
        "population_scaled": "Population (0–100)",
        "housing_cost_burden_pct": "Renters ≥30% income on rent",
        "housing_cost_burden_scaled": "Cost burden (0–100)",
    },
    inplace=True,
)

score_df["Median income"] = score_df["Median income"].apply(fmt_currency)
score_df["Median rent"] = score_df["Median rent"].apply(fmt_currency)
score_df["Median home value"] = score_df["Median home value"].apply(fmt_currency)
score_df["Population"] = score_df["Population"].apply(fmt_int)
score_df["% bachelor+"] = score_df["% bachelor+"].apply(fmt_percent)
score_df["Poverty rate"] = score_df["Poverty rate"].apply(fmt_percent)
score_df["Unemployment rate"] = score_df["Unemployment rate"].apply(fmt_percent)
score_df["Renters ≥30% income on rent"] = score_df[
    "Renters ≥30% income on rent"
].apply(fmt_percent)

st.dataframe(
    score_df.sort_values("Score", ascending=False),
    use_container_width=True,
)

# ========= PROJECT 2 REPORT (DETAILED VERSION) =========
st.header("Project 2 Report: Regional ZIP Code Analysis")

st.markdown(
    """
### 1. Introduction

This dashboard was developed as part of **Project 2: Regional Data Visualization**, which requires the
collection, analysis, scoring, and visualization of ZIP code–level regional data. The project aims to
evaluate differences in socioeconomic, demographic, and housing conditions within a selected county
and present the results through an interactive dashboard.

For this project, the selected region is:

- **State:** New York  
- **County:** Suffolk County  
- **Geographic Unit:** ZIP Code Tabulation Areas (ZCTAs)  

Suffolk County is a large, economically diverse suburban region, making it an ideal case study  
for comparing ZIP code conditions and computing an aggregated composite score that reflects  
overall desirability and livability.

---

### 2. Data Collection & Sources

To satisfy project requirements, **eight variables** were collected for every ZIP code in Suffolk County.
All data used in this study were obtained exclusively from **public, authoritative, government sources**, 
primarily the **U.S. Census Bureau’s American Community Survey (ACS) 2019–2023 5-year estimates**.

Data collection followed these steps:

1. Identified relevant ACS tables that provide ZIP-code-level socioeconomic and housing indicators.  
2. Queried the data using the official **Census Data API** (https://api.census.gov/).  
3. Gathered ZIP boundary geometries using official **U.S. Census ZCTA GeoJSON** for Suffolk County.  
4. Cleaned, validated, and merged the datasets using ZIP/ZCTA codes as join keys.  
5. Handled missing or suppressed ACS values (negative placeholders) by converting them into `NaN`  
   for accurate scoring and plotting.

All numeric transformations, normalizations, and scoring calculations were performed in Python  
and exported to a final cleaned dataset included in the dashboard.

---

### 3. Variables Included in the Analysis

The dashboard uses **eight key variables**, exceeding the minimum requirement of five. These variables were 
chosen because they represent major dimensions of economic opportunity, housing affordability, 
educational attainment, population structure, and financial stress.

1. **Median household income** (ACS Table B19013)  
2. **% of adults 25+ with a bachelor’s degree or higher** (ACS Table B15003)  
3. **Poverty rate** (ACS Table B17001)  
4. **Unemployment rate** (ACS Table B23025)  
5. **Median gross rent** (ACS Table B25064)  
6. **Median home value** (ACS Table B25077)  
7. **Total population** (ACS Table B01003)  
8. **% of renter households spending ≥30% of income on rent (housing cost burden)** (ACS Table B25070)  

Together, these variables provide a multidimensional view of ZIP-level conditions.

---

### 4. Data Processing & Cleaning Procedures

Several data processing steps were implemented to ensure accuracy:

- Negative ACS suppression codes were replaced with `NaN`.  
- All numeric fields were converted to appropriate numeric types.  
- Missing values were handled carefully during scaling to avoid distortions.  
- Scaling transformations were applied only within the Suffolk County ZIP set.  
- Final CSV output was standardized and validated before use in visualizations.

---

### 5. Composite Score Methodology

The goal of the composite score is to summarize each ZIP code's overall desirability based on  
multiple quantitative indicators. The methodology follows a **two-step process**:

#### Step 1 — Normalization (0–100 scoring)

All variables were normalized using **min–max scaling**, which rescales values within the observed range:

- **Higher-is-better:** income, education, home value, population  
- **Lower-is-better:** poverty rate, unemployment rate, rent, housing cost burden  

For lower-is-better variables, the scale is reversed so that lower raw values produce higher normalized scores.

#### Step 2 — Weighted Composite Score

Each variable contributes to the final score based on its relative importance:

- Median household income — **20%**  
- Educational attainment — **15%**  
- Median home value — **15%**  
- Poverty rate — **10%**  
- Unemployment rate — **10%**  
- Median gross rent — **10%**  
- Population — **10%**  
- Housing cost burden — **10%**  

The final **composite_score** ranges from **0 to 100**, where higher values represent ZIP codes with  
more favorable characteristics across the selected indicators.

---

### 6. Dashboard Design & Visualization Features

The dashboard includes several interactive components:

- **Choropleth map** of ZIP composite scores  
- **Variable comparison bar chart**  
- **Income vs. poverty scatter plot** (bubble = population, color = composite score)  
- **Full score breakdown table** with raw values and normalized scores  
- **Sidebar filters** (ZIP prefix, income range, population range)  

These features together support exploratory analysis and highlight geographic patterns.

---

### 7. APA Data Citations

All data sources used in this project are cited in APA format in the **Data Sources** section of the dashboard.

This includes ACS tables:

- B19013 — Median household income  
- B15003 — Educational attainment  
- B17001 — Poverty status  
- B23025 — Employment status  
- B25064 — Median gross rent  
- B25077 — Median home value  
- B01003 — Total population  
- B25070 — Housing cost burden  

as well as official Census ZCTA boundary data for Suffolk County.
"""
)

# ========= DATA SOURCES =========
st.subheader("Data sources and References")

st.markdown(
    """
- U.S. Census Bureau. (2024). *2019–2023 American Community Survey 5-year estimates, Table B19013: Median household income in the past 12 months (in 2023 inflation-adjusted dollars).* Retrieved from https://api.census.gov/
- U.S. Census Bureau. (2024). *2019–2023 American Community Survey 5-year estimates, Table B15003: Educational attainment for the population 25 years and over.* Retrieved from https://api.census.gov/
- U.S. Census Bureau. (2024). *2019–2023 American Community Survey 5-year estimates, Table B17001: Poverty status in the past 12 months.* Retrieved from https://api.census.gov/
- U.S. Census Bureau. (2024). *2019–2023 American Community Survey 5-year estimates, Table B23025: Employment status for the population 16 years and over.* Retrieved from https://api.census.gov/
- U.S. Census Bureau. (2024). *2019–2023 American Community Survey 5-year estimates, Tables B25064 & B25077: Median gross rent and median home value.* Retrieved from https://api.census.gov/
- U.S. Census Bureau. (2023). *ZIP Code Tabulation Areas (ZCTAs): Geographic Terms and Concepts.* Retrieved from https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html
- U.S. Census Bureau. (2023). *TIGER/Line Shapefiles: ZIP Code Tabulation Areas (ZCTAs).* Retrieved from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- OECD. (2008). *Handbook on Constructing Composite Indicators: Methodology and User Guide.* Organisation for Economic Co-operation and Development.  
  Retrieved from https://www.oecd.org/sdd/42495745.pdf
"""
)


# ========= DOWNLOAD PDF REPORT =========
st.header("Download Full Project Report")

final_pdf_text = f"""
Suffolk County ZIP Code Analysis – Full Project Report

1. Project Overview
This project evaluates ZIP code socioeconomic and housing conditions in Suffolk County, NY.
A composite desirability score is computed using eight ACS variables.

2. Variables Included
- Median household income
- Educational attainment (% bachelor’s degree or higher)
- Poverty rate
- Unemployment rate
- Median gross rent
- Median home value
- Total population
- Renter housing cost burden (% spending ≥30% of income)

3. Composite Score Methodology
All variables are normalized using 0–100 min–max scaling. Higher-is-better and lower-is-better
variables are reversed as needed. Weighting scheme:
Income (20%), Education (15%), Home value (15%), Poverty (10%), Unemployment (10%),
Median rent (10%), Population (10%), Cost burden (10%).

4. Key Insights & Findings (current filter view)
{key_insights_md.replace("*", "").replace("**", "")}

5. Data Sources
2019–2023 ACS 5-year tables: B19013, B15003, B17001, B23025, B25064, B25077, B01003, B25070.
"""

pdf_buffer = generate_pdf(final_pdf_text)

st.download_button(
    label="📄 Download PDF Report",
    data=pdf_buffer,
    file_name="Suffolk_County_ZIP_Report.pdf",
    mime="application/pdf",
)
