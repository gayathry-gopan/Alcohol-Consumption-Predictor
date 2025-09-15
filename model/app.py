import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to add a blurred background image and custom CSS
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background: transparent !important;
        }
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url("https://ik.imagekit.io/drinksvine/blog/types-of-spirits.webp");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: blur(3px);
            z-index: -2;
        }
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: -1;
        }

        /* Keep Streamlit content ABOVE background */
        .stApp .main .block-container {
            position: relative;
            z-index: 0;
        }
        
        /* Custom CSS to change the container border to #FF4B4B */
        div.st-emotion-cache-z5f8h3.e1f1d6gn5 {
            border: 5px solid #FF4B4B !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Call the function to set the background
add_bg_from_url()

# Use Streamlit's caching to avoid reloading data
@st.cache_data
def load_and_train_model():
    """Loads the dataset, trains the model, and returns the trained model and features."""
    try:
        df = pd.read_csv('model/beer-servings.csv')
    except FileNotFoundError:
        st.error("Error: 'beer-servings.csv' not found. Please ensure the file is in the same directory.")
        return None, None
    
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    features = ['beer_servings', 'spirit_servings', 'wine_servings']
    target = 'total_litres_of_pure_alcohol'
    
    X = df[features]
    y = df[target]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, features

# Load the model and features
model, features = load_and_train_model()

# --- Streamlit App UI ---
if model:
    st.title(' Alcohol Consumption Predictor')
    st.markdown('A machine learning model for predicting total alcohol intake based on serving data.')
    
    with st.container(border=True):
        st.subheader('Prediction Parameter')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            beer_servings = st.number_input('Beer Servings', min_value=0, value=0)
        with col2:
            spirit_servings = st.number_input('Spirit Servings', min_value=0, value=0)
        with col3:
            wine_servings = st.number_input('Wine Servings', min_value=0, value=0)
    
    st.markdown("---")
    
    if st.button('Predict Total Alcohol', type='primary'):
        input_data = pd.DataFrame([[beer_servings, spirit_servings, wine_servings]], columns=features)
        
        prediction = model.predict(input_data)[0]
        
        st.subheader('Prediction Result')
        st.success(f'Predicted total liters of pure alcohol: **{prediction:.2f}**')

# Close main-content wrapper
st.markdown('</div>', unsafe_allow_html=True)


