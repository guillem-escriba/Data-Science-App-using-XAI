import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
page_title="Price Prediction",
page_icon="💰",
layout="wide",
initial_sidebar_state="expanded")

#Load the pickle file with the model and the label encoders
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

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

df_original = load_data()

st.title("🔮 Car price Predictor 🔮")
st.write("""### Enter your car information to predict its price!""")

car = st.text_input('Car brand', value="Porshe")

body_types = (
    "crossover",
    "sedan",
    "van",
    "vagon",
    "hatch",
    "other",
)

body = st.selectbox("Body", body_types)

mileage = st.slider("Mileage", 0, 600, 80)

engV = st.slider("EngV", 0.0, 7.0, 3.5)

engType_types = (
    "Gas",
    "Petrol",
    "Diesel",
    "Other",
)
engType = st.selectbox("EngType", engType_types)

registered = st.radio(
    "Is it registered?",
    ('Yes', 'No'))

year  = st.slider("Year", 1975, 2015, 2010)

drive_types = (
    "full",
    "rear",
    "front",
)
drive = st.selectbox("Drive", drive_types)

yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']


ok = st.button("Calculate Price")
if ok:
    X_sample = np.array([[car, body, mileage, engV, engType, registered, year, drive ]])
    # Apply the encoder and data type corrections:
    X_sample[:, 0] = str(X_sample[:, 0][0] if X_sample[:, 0][0] in list(df_original['car'].unique()) else 'Other')
    X_sample[:, 0] = le_car.transform(X_sample[:,0])
    X_sample[:, 1] = le_body.transform(X_sample[:,1])
    X_sample[:, 4] = le_engType.transform(X_sample[:,4])
    X_sample[:, 5] = int(1 if X_sample[:, 5][0] in yes_l else 0)
    X_sample[:, 7] = le_drive.transform(X_sample[:,7])

    X_sample = np.array([[
        int(X_sample[0, 0]), 
        int(X_sample[0, 1]), 
        int(X_sample[0, 2]), 
        float(X_sample[0, 3]), 
        int(X_sample[0, 4]), 
        int(X_sample[0, 5]), 
        int(X_sample[0, 6]), 
        int(X_sample[0, 7])
    ]])
   
    price = model.predict(X_sample)
    st.subheader(f"The estimated price is ${price[0]:.2f}")

    # Set a dark background style for the plot
    plt.style.use('dark_background')

    current_year = year
    # Generate predictions
    year_range = np.arange(current_year, current_year + 11) 
    years_old = np.arange(current_year-10, current_year+1) 
    years_old = sorted(years_old, reverse=True)
    predicted_prices = []
    for year in years_old:
        X_sample[0,6] = int(year)
        predicted_prices.append(model.predict(X_sample)[0])

    # Create a DataFrame
    price_data = pd.DataFrame({
        'Year': year_range,
        'Predicted Price': predicted_prices
    })

    # Identify min and max prices
    min_price = price_data['Predicted Price'].min()
    max_price = price_data['Predicted Price'].max()
    min_year = price_data[price_data['Predicted Price'] == min_price]['Year'].values[0]
    max_year = price_data[price_data['Predicted Price'] == max_price]['Year'].values[0]

    # Plot using Matplotlib and Seaborn
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Matplotlib line plot for predictions
    ax1.plot(price_data['Year'], price_data['Predicted Price'], color='cyan', marker='o', label='Predicted Price')

    # Seaborn regression plot for trend line
    sns.regplot(x='Year', y='Predicted Price', data=price_data, scatter=False, color='yellow', label='Trend Line', ax=ax1)

    # Highlighting min and max points
    ax1.scatter([min_year, max_year], [min_price, max_price], color='yellow', zorder=5)

    # Adding labels for min and max
    ax1.text(min_year, min_price, f' Min: ${min_price:.2f}', color='yellow', ha='right')
    ax1.text(max_year, max_price, f' Max: ${max_price:.2f}', color='yellow', ha='right')

    ax1.set_title("Predicted Car Price vs. Years", color='white')
    ax1.set_xlabel("Year", color='white')
    ax1.set_ylabel("Predicted Car Price", color='white')
    ax1.legend()
    ax1.grid(True, color='gray')

    # Display the plot in Streamlit
    st.pyplot(fig)