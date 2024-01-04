from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import numpy as np

#Load the pickle file with the model and the label encoders
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Explaniability",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:guillem.escriba01@estudiant.upf.edu',
        'Report a bug': "mailto:guillem.escriba01@estudiant.upf.edu",
        'About': "# This is a *Car Price Prediction App* to assist in pricing vehicles accurately and fairly."
    }
)

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

@st.cache_data
def load_data():
    df =  pd.read_csv("car_ad_display.csv", encoding = "ISO-8859-1", sep=";").drop(columns='Unnamed: 0')
    df = df.dropna()
    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[df["price"] <= 100000]
    df = df[df["price"] >= 1000]
    df = df[df["mileage"] <= 600]
    df = df[df["engV"] <= 7.5]
    df = df[df["year"] >= 1975]

    return df

data = load_model()
model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

df = load_data()

df['car'] = le_car.transform(df['car'])
df['body'] = le_body.transform(df['body'])
df['engType'] = le_engType.transform(df['engType'])
df['drive'] = le_drive.transform(df['drive'])

df = df.drop(columns='model')
yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']
df['registration'] = np.where(df['registration'].isin(yes_l), 1, 0)


# Split data for Shap explanations
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Display the test set for reference
st.write("Test Dataset for reference:")
st.dataframe(X_test)

# Summary Plot
st.subheader("Global Feature Importance")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Bar Plot
st.subheader("Feature Importance Bar Plot")
fig = shap.plots.bar(shap_values, show=False)
st.pyplot(fig)

# Scatter Plots for individual features
st.subheader("SHAP Value Scatter Plot - Mileage")
fig, ax = plt.subplots()
shap.plots.scatter(shap_values[:, "mileage"], ax=ax)
st.pyplot(fig)

st.subheader("SHAP Value Scatter Plot - Engine Volume (engV)")
fig, ax = plt.subplots()
shap.plots.scatter(shap_values[:, "engV"], ax=ax)
st.pyplot(fig)

st.subheader("SHAP Value Scatter Plot - Year")
fig, ax = plt.subplots()
shap.plots.scatter(shap_values[:, "year"], ax=ax)
st.pyplot(fig)

# Waterfall Plot for the first sample
st.subheader("Waterfall Plot for First Sample")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# Force Plot for the first sample
st.subheader("Force Plot for First Sample")
shap.force_plot(shap_values[0].base_values, shap_values[0].values, X_test.iloc[0], matplotlib=True)
st.pyplot(bbox_inches='tight')

# Decision Plot for the first sample
st.subheader("Decision Plot for First Sample")
fig, ax = plt.subplots()
shap.decision_plot(shap_values[0].base_values, shap_values[0].values, X_test.iloc[0], show=False)
st.pyplot(fig)