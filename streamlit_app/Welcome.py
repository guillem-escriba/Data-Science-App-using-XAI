import streamlit as st

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:guillem.escriba01@estudiant.upf.edu',
        'Report a bug': "mailto:guillem.escriba01@estudiant.upf.edu",
        'About': "# This is a *Car Price Prediction App* to assist in pricing vehicles accurately and fairly."
    }
)

# Home Page
st.title('🚗 Car Price Prediction Home Page')

# Application Description
st.markdown("""
Welcome to the Car Price Prediction platform, the interactive web application designed to provide data-driven price estimations for vehicle sellers and buyers. 
Our application offers insightful analysis of car features and their impact on market value, facilitating informed decision-making.
""")
st.sidebar.header("Navigate through the different tabs to learn about all the features of this app")
st.sidebar.write(" 📢 In the welcome tab we will find a brief introduction to the use case: Vehicle Pricing. ")

st.write("""We are one of the most popular car buying and selling platforms in the world. We are going to launch a new product based on a price recommender 
         for users' vehicles. In this application you will be able to explore the data of vehicles advertised in the past, test the prediction model, and 
         understand the model's decisions with the explainability tab.""")




# The Goal of the App
st.header('🎯 The Goal of the App')
st.write("""
The main objectives of this application are to:

- Enable users to understand the factors that influence car valuations through interactive data visualizations.
- Provide an accurate predictive tool that estimates car prices using a machine learning model trained on historical sales data.
""")

# Explanation of the Dataset
st.header('📖 About the Dataset')
st.write("""
The dataset underpinning this application comes from a compilation of car sales records, including attributes such as:

- Make and model of the car
- Body type, engine type, and drive type
- Mileage, engine volume, and year of manufacture
- Registration status

This dataset allows for a granular analysis of car prices based on a myriad of factors.
""")

# Additional Comments
st.header('💡 Additional Comments')
st.write("""
Please consider the following while using the app:

- The price predictions are generated based on historical data which may not fully reflect the current market dynamics.
- The provided data visualizations aim to simplify the understanding of complex relationships between car features and prices.
- Users are encouraged to experiment with different inputs to gain insights into the valuation of their vehicles.
""")

# Footer
st.markdown("---")
st.subheader('Get Started')
st.write("Use the navigation bar to switch between the 'Price Prediction', 'Data Exploration', and 'Model Explanation' sections to start your journey.")

