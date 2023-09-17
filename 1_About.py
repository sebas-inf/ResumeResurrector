

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras import add_vertical_space as avs
from langchain.callbacks import get_openai_callback
import base64
import os
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from design import toggle
toggle()


import requests
import streamlit as st 
from streamlit_lottie import st_lottie

with st.container():
    left_column,right_column = st.columns(2)
    with left_column:
        st.header("Resume Resurrection")
        st.divider()
        st.write("About")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
    
lottie_coding = load_lottieurl("https://lottie.host/71a49c0d-c96c-41c6-afb7-434adbd8b01c/AbDotMP4M9.json")

with right_column:
    st_lottie(lottie_coding,height =200,key="paperplane")

import streamlit as st

# Define a function to toggle between dark and light modes
def toggle():
    on = st.checkbox('Activate Light Mode')
    if on:
        st.set_page_config(
            page_title="Your App Title",
            page_icon="✅",
            layout="centered",
            primaryColor="#F63366",              # Dark mode primary color
            backgroundColor="#FFFFFF",           # White background color
            secondaryBackgroundColor="#F0F2F6",  # Secondary background color
            textColor="#262730",                # Text color
            font="sans serif",
        )
    else:
        st.set_page_config(
            page_title="Your App Title",
            page_icon="✅",
            layout="centered",
            primaryColor="#F63366",              # Dark mode primary color
            backgroundColor="#262730",           # Dark mode background color
            secondaryBackgroundColor="#1E2125",  # Secondary background color
            textColor="#FFFFFF",                # Text color in dark mode
            font="sans serif",
        )

# Set the initial theme (dark mode by default)
st.set_page_config(
    page_title="Your App Title",
    page_icon="✅",
    layout="centered",
    primaryColor="#F63366",              # Dark mode primary color
    backgroundColor="#262730",           # Dark mode background color
    secondaryBackgroundColor="#1E2125",  # Secondary background color
    textColor="#FFFFFF",                # Text color in dark mode
    font="sans serif",
)

# Toggle between dark and light modes
toggle()

# Rest of your Streamlit app content
st.write("ABOUT ")
with st.container():
    st.write("")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Resume Resurrection")
        st.write("About")
