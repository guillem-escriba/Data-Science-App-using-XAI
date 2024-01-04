import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dynamic Visualization",
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

# Function to generate boxplot using Plotly Express
def generate_boxplot(dataframe, x, y, title):
    fig = px.box(dataframe, x=x, y=y, title=title, color=x)
    fig.update_layout(showlegend=False)
    return fig


# Load the datasets
df = load_data()

# Define sections
sections = [ 'Interactive Distribution Histogram per brands', 
            'Boxplot of Price Distribution by Car Brand', 'Boxplot of Price Distribution by Body Type', 
            'Price vs Mileage', 'Interactive Bubble Chart: Price and Mileage by Car Brand']

# Sidebar for navigation
st.sidebar.title('Navigation')
selected_section = st.sidebar.radio('Go to', sections)



if selected_section == 'Interactive Distribution Histogram per brands':
    # Dropdown to select Car Brand
    car_brand = st.multiselect('Select Car Brand', options=df['car'].unique(), default=df['car'].unique())
    
    # Filtering data based on selection
    column = st.selectbox('Select a feature to plot', options=df.columns[1:])
    filtered_df = df[df['car'].isin(car_brand)]
    
    # Interactive Histogram for Price Distribution
    st.subheader(f'Interactive {column} Distribution Histogram per Brand')
    fig = px.histogram(filtered_df, x=column, color='car', nbins=50, title=f'{column} Distribution by Brand')
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)

if selected_section == 'Boxplot of Price Distribution by Car Brand':
    st.subheader("Boxplot of Price Distribution by Car Brand")
    fig = generate_boxplot(df, x='car', y='price', title="Price Distribution by Car Brand")
    st.plotly_chart(fig)

if selected_section == 'Boxplot of Price Distribution by Body Type':
    st.subheader("Boxplot of Price Distribution by Body Type")
    fig = generate_boxplot(df, x='body', y='price', title="Price Distribution by Body Type")
    st.plotly_chart(fig)

if selected_section == 'Price vs Mileage':
    st.subheader("Price vs Mileage")
    fig = px.scatter(df, x='mileage', y='price', color='car', title='Price vs. Mileage', hover_data=['car', 'year'])
    st.plotly_chart(fig)

if selected_section == 'Interactive Bubble Chart: Price and Mileage by Car Brand':
    st.subheader("Interactive Bubble Chart: Price and Mileage by Car Brand")
    # We can assume the count is needed, similar to the original code
    df['count'] = df.groupby(['car', 'price', 'mileage'])['car'].transform('count')
    fig = px.scatter(df, x='mileage', y='price', size='count', color='car', hover_name='car', title='Price and Mileage by Car Brand')
    st.plotly_chart(fig)
if selected_section == 'Price vs Year':
    st.subheader("Price vs Year of Manufacture")
    # Slider for selecting the range of years
    year_range = st.slider('Select the range of years', int(df['year'].min()), int(df['year'].max()), (int(df['year'].min()), int(df['year'].max())))
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Scatter plot for Price vs Year
    fig = px.scatter(filtered_df, x='year', y='price', color='car', title='Price vs. Year of Manufacture', hover_data=['car', 'mileage'])
    st.plotly_chart(fig)

if selected_section == 'Engine Volume Distribution':
    st.subheader("Distribution of Engine Volume")
    # Dropdown to select Engine Type
    engine_type = st.multiselect('Select Engine Type', options=df['engType'].unique(), default=df['engType'].unique())
    
    # Filtering data based on selection
    filtered_df = df[df['engType'].isin(engine_type)]
    
    # Histogram for Engine Volume Distribution
    fig = px.histogram(filtered_df, x='engV', nbins=50, title='Engine Volume Distribution', color='engType')
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)

if selected_section == 'Correlation Heatmap':
    st.subheader("Correlation Heatmap of Features")
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Heatmap for the correlation matrix
    fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r', title='Feature Correlation Heatmap')
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.subheader('Discover Insights')
st.write("Use the navigation bar to explore different visualizations and uncover insights into car pricing trends.")
