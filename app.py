import streamlit as st
import streamlit.components.v1 as components
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from pathlib import Path
import streamlit_authenticator as stauth
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import numpy as np


# Load environment variables from .env file
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

def sign_up_user(email, password):
    try:
        user = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return user
    except Exception as e:
        st.error(f"Error signing up: {e}")
        return None

def sign_in_user(email, password):
    try:
        user = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return user
    except Exception as e:
        st.error(f"Error signing in: {e}")
        return None
def sign_out_user():
    try:
        # Server-side sign out
        supabase.auth.sign_out()

        # Clear all user/session info in Streamlit
        # IMPORTANT: Delete the correct key from Streamlit's session state
        if 'user_email' in st.session_state:
            del st.session_state['user_email']

        # Force rerun so UI updates immediately
        st.rerun()
        auth_screen()
    except Exception as e:
        st.error(f"Error signing out: {e}")

def auth_screen():
    st.title('Depression Detection App')
    
    option=st.selectbox('Select an option', key="auth_option",options=['Sign In', 'Sign Up'])
    email=st.text_input('Email',key="auth_email")
    password=st.text_input('Password', type='password',key="auth_password")

    if option=='Sign Up' and st.button('Register',key='signup_button'):
        user=sign_up_user(email, password)
        if user and user.user:
            st.success('Sign up successful! Please sign in.')

    if option=='Sign In' and st.button('Login',key='login_button'):
        user=sign_in_user(email, password)
        if user and user.user:
            st.session_state.user_email=user.user.email
            st.success('Sign in successful {email}!')
            st.rerun()




# if 'user_email' not in st.session_state:
#     st.session_state.user_email=None

# if not st.session_state.user_email:
#     # Only show authentication screen if user not logged in
#     st.sidebar.info("🔒 Please sign in to access the app.")
#     auth_screen()
# else:
#     # Show sidebar navigation once logged in
#     st.sidebar.success(f"✅ Logged in as {st.session_state.user_email}")

#     # Define accessible pages
#     page_names_to_funcs = {
#         "Introduction": intro_page,
#         "Detection ML based": main_page_ml,
#         "Detection DL based": main_page_dl
#     }

#     # Sidebar selection
#     demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys(), key="page_select")
#     page_names_to_funcs[demo_name]()

#     # Logout button
#     if st.sidebar.button("🚪 Sign Out"):
#         sign_out_user()
#         st.rerun()


# Initialize session state at the very top of your script
#if "user_email" not in st.session_state:
 #   st.session_state.user_email = None

# Now, the rest of your app can safely check the value
#if not st.session_state.user_email:
    # User is not logged in, show the login screen
 #   auth_screen()
#else:
    # User is logged in, show the main app
 #   st.sidebar.success(f"✅ Logged in as {st.session_state.user_email}")
    # ... rest of your logged-in logic



def intro_page():
    #some information about depression 
    components.html(
        """
        <div style="
            font-family: 'Segoe UI', sans-serif; 
            padding: 30px; 
            border-radius: 12px; 
            background: linear-gradient(to right, #f8f9fa, #f1f1f1); 
            box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
            margin: 20px auto; 
            width: 85%;
        ">
            <h1 style="
                text-align: center; 
                color: #222; 
                font-size: 36px; 
                margin-bottom: 10px;
            ">
                🌿 Depression Detection levergaing Natural Language Processing and Machine Learning
            </h1>
            <p style="
                text-align: center; 
                font-size: 18px; 
                color: #555; 
                margin-bottom: 25px;
            ">
                Using Natural Language Processing (NLP) & Machine Learning (ML) to support early detection and awareness
            </p>

            <div style="margin-bottom: 25px;">
                <h2 style="color: #0077b6; font-size: 24px;">💡 What is Depression?</h2>
                <p style="font-size: 16px; color: #444; line-height: 1.6;">
                    Depression is a common but serious mental health disorder that negatively affects how a person feels, thinks, and behaves. 
                    It can lead to emotional and physical problems, reducing the ability to function effectively in daily life. 
                    According to the World Health Organization, millions of people worldwide are affected by depression, making it one of the leading causes of disability.
                </p>
            </div>

            <div>
                <h2 style="color: #0096c7; font-size: 24px;">🤖 Why NLP & Machine Learning?</h2>
                <p style="font-size: 16px; color: #444; line-height: 1.6;">
                    In today’s digital age, people express their emotions through texts, social media posts, and online conversations. 
                    <b>Natural Language Processing (NLP)</b> allows computers to understand and analyze these textual patterns, 
                    while <b>Machine Learning (ML)</b> helps in building predictive models that can identify signs of depression. 
                    Together, they provide a data-driven way to assist in early detection, awareness, and support systems for mental health.
                </p>
            </div>
        </div>
        """,
        height=800
    )



def main_page_ml():
    st.title('Sentia - Depression Detection App')
    st.markdown(
    """
    <div style="
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        background: linear-gradient(to right, #f8f9fa, #eaeaea);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    ">
        <h2 style="color: #222; margin-bottom: 10px;">
            👋 Welcome to <span style="color:#0077b6;">Sentia</span>
        </h2>
        <p style="color: #444; font-size: 16px; line-height: 1.6;">
            A depression detection app leveraging <b>Natural Language Processing (NLP)</b>, 
            <b>Machine Learning (ML)</b>, and <b>Deep Learning (DL)</b> 
            to predict emotions from text and support early awareness.
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
    """
    <h3 style='
        text-align: center; 
        color: #0077b6; 
        font-family: "Segoe UI", sans-serif; 
        margin-bottom: 10px;
    '>
        ✍️ Enter Text Below
    </h3>
    """,
    unsafe_allow_html=True
    )

    # Custom CSS for input box
    st.markdown(
    """
    <style>
    div[data-baseweb="input"] {
        width: 100% !important;      /* stretch full width */
    }
    div[data-baseweb="input"] > div {
        font-size: 18px !important;  /* bigger text */
        padding: 12px !important;    /* spacing inside */
        border-radius: 10px !important; /* rounded corners */
        border: 2px solid #0077b6 !important; /* blue border */
        box-shadow: 1px 1px 6px rgba(0,0,0,0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True  
    )
     #taking the input from user
    text=st.text_area(label='',
                      max_chars=200,
                      placeholder='Enter text',
                      height=100)
    

    #loading the models and vectorizer
    model=pickle.load(open('model.pkl','rb'))
    vectorizer=pickle.load(open('tfidf.pkl','rb'))

    #converting text to vector and making prediction
    if text is not None and text!='':
        #making prediction
        vectorized_text=vectorizer.transform([text])
        result=model.predict(vectorized_text)[0]

        if result==1:
            st.warning('The text shows some signs of depression')
        else:
            st.success('The text shows no signs of depression')
    else:
        st.warning('Please enter some text to predict')

#for future use DEEP LEARNING RNN BASED
def main_page_dl():
    st.title('Sentia - Depression Detection App')
    st.markdown(
    """
    <div style="
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        background: linear-gradient(to right, #f8f9fa, #eaeaea);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    ">
        <h2 style="color: #222; margin-bottom: 10px;">
            👋 Welcome to <span style="color:#0077b6;">Sentia</span>
        </h2>
        <p style="color: #444; font-size: 16px; line-height: 1.6;">
            A depression detection app leveraging <b>Natural Language Processing (NLP)</b>, 
            <b>Machine Learning (ML)</b>, and <b>Deep Learning (DL)</b> 
            to predict emotions from text and support early awareness.

            This page uses deep learning RNN based model
       
    </div>
    """,
    unsafe_allow_html=True
    )

    #RNN image 
    st.image('rnn.webp',use_container_width=True)
    st.markdown(
    """
    <h3 style='
        text-align: center; 
        color: #0077b6; 
        font-family: "Segoe UI", sans-serif; 
        margin-bottom: 10px;
    '>
        ✍️ Enter Text Below
    </h3>
    """,
    unsafe_allow_html=True
    )

    # Custom CSS for input box
    st.markdown(
    """
    <style>
    div[data-baseweb="input"] {
        width: 100% !important;      /* stretch full width */
    }
    div[data-baseweb="input"] > div {
        font-size: 18px !important;  /* bigger text */
        padding: 12px !important;    /* spacing inside */
        border-radius: 10px !important; /* rounded corners */
        border: 2px solid #0077b6 !important; /* blue border */
        box-shadow: 1px 1px 6px rgba(0,0,0,0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True  
    )
     #taking the input from user
    text=st.text_area(label='',
                      max_chars=200,
                      placeholder='Enter text',
                      height=100)

    #loading the models and vectorizer
    # Load the saved tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the Keras model correctly
    model = tf.keras.models.load_model('rnn_fastext.h5')
    seq=tokenizer.texts_to_sequences([text])
    padded=np.array(tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=500))

    #converting text to vector and making prediction
    if text is not None and text!='':
        #making prediction
        result=model.predict(padded)[0][0]

        if result>0.5:
            st.warning('The text shows some signs of depression with a probability of '+str(round(result*100,2))+'%')
        else:
            st.success('The text shows no signs of depression')
    else:
        st.warning('Please enter some text to predict')


# 1. Initialize session state
if 'user_email' not in st.session_state:
    st.session_state.user_email = None

# Main control flow
if not st.session_state.user_email:
    # If user is not logged in, show the authentication screen
    auth_screen()
else:
    # If user is logged in, show the main application
    st.sidebar.success(f"✅ Logged in as {st.session_state.user_email}")

     #Define pages for logged-in users
    page_names_to_funcs = {
        "Introduction": intro_page,
        "Detection ML based": main_page_ml,
        "Detection DL based": main_page_dl
    }

    # Sidebar page selection
    selected_page = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

    # Logout button
    if st.sidebar.button("🚪 Sign Out"):
        sign_out_user()
        st.rerun()


#demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
#page_names_to_funcs[demo_name]()