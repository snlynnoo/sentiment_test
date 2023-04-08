from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump,load
import streamlit as st
# import helper

st.set_page_config(page_title="News Summarizer", page_icon="📜", layout="wide")

hide_menu_style = """
    <style>
        #MainMenu {visibility : hidden}
        footer {visibility : hidden}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)
 # Step 6: Deploy the model
loaded_model = load('random_forest_model.joblib')

with st.form(key='values'):
    st.write("📰 Input tweets to classify")
    new_data = st.text_area("📰 Input News to Summarize", label_visibility = "collapsed")
    
    submitted_data = st.form_submit_button(label = 'Predict')

    if submitted_data:
        new_data_prediction = loaded_model.predict([new_data])

 # Interpretation for model result
        if new_data_prediction == -1:
            st.error("Negative")
        elif new_data_prediction == 0:
            st.success("Neutral")
        else:
            st.success("Positive")
