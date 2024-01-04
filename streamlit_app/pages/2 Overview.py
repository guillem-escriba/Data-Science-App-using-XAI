import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px

st.set_page_config(
    page_title="Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:guillem.escriba01@estudiant.upf.edu',
        'Report a bug': "mailto:guillem.escriba01@estudiant.upf.edu",
        'About': "# This is a *Car Price Prediction App* to assist in pricing vehicles accurately and fairly."
    }
)


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

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# Load the datasets
df = load_data()

# Set the aesthetic style of the plots
plt.style.use('dark_background')

# Sections of the menu
sections = ['Data Overview', 'Distribution by Car Brand', 'Average Price by Body Type', 
            'Price Distribution by Mileage', 'Price Distribution by Year']

# Sidebar for navigation
st.sidebar.title('Navigation')
selected_section = st.sidebar.radio('Go to', sections)

# Conditional rendering based on selected section
if selected_section == 'Data Overview':
    st.title('Car Price Data Overview')
    st.write('Please select a section from the sidebar to start exploring the data.')
    st.header("Overview of the Car Advertisements")
    if st.checkbox("Show Data Summary"):
        st.write("Here you can explore the dataset used for the model:")
        st.dataframe(df.head(10), width=1500, height=600)
        st.write("The numerical data has the following statistics:")
        st.write(df.describe())

if selected_section == 'Distribution by Car Brand':
    st.subheader("Distribution by Car Brand")
    brand_counts = df['car'].value_counts().head(20)
    fig, ax = plt.subplots()
    ax.bar(brand_counts.index, brand_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Car Brand')
    plt.ylabel('Number of Ads')
    plt.title('Top 20 Car Brands by Number of Ads')
    st.pyplot(fig)

if selected_section == 'Average Price by Body Type':
    st.subheader("Average Price by Body Type")
    avg_price_body = df.groupby('body')['price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=avg_price_body.values, y=avg_price_body.index, ax=ax)
    plt.xlabel('Average Price')
    plt.ylabel('Body Type')
    plt.title('Average Price by Body Type')
    st.pyplot(fig)

if selected_section == 'Price Distribution by Mileage':
    st.subheader("Price Distribution by Mileage")
    fig, ax = plt.subplots()
    sns.scatterplot(x='mileage', y='price', data=df, ax=ax)
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Price Distribution by Mileage')
    st.pyplot(fig)

if selected_section == 'Price Distribution by Year':
    st.subheader("Price Distribution by Year")
    avg_price_year = df.groupby('year')['price'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(x='year', y='price', data=avg_price_year, ax=ax)
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.title('Average Price by Year')
    st.pyplot(fig)