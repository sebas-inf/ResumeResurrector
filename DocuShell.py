#Main
import streamlit as st
import openai
from dotenv import load_dotenv
import pickle
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

st.set_page_config(page_title="Resume Reviewer", page_icon="📖")


with st.sidebar:
    st.title("Resume Reviewer")
    st.markdown('''
    ## About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

 
    ''')
    avs.add_vertical_space(5)
    st.write('Made by Spanish Indian Inquision')


def show_pdf(file_url):
    pdf_embed_code = f'<iframe src="{file_url}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_embed_code, unsafe_allow_html=True)

def show_content(book):
    if book == "Matthew":
        file_id = '1PBMShoyok8oO-elmF55KW4gB6cy4yhxY'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Mark":
        file_id = '1Dvn-CfxwrDJRGMOe83DbcqZU83XvpX1d'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Luke":
        file_id = '1utUYxshKZlo2xdlPeXkT6u9MU4Y-8sXb'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "John":
        file_id = '1WVpTHioNAgdpPkTSZ8_29rxK6QDJi_S5'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Acts":
        file_id = '1A2XVLo32sSPSdfHSNCGMoavq2Ryn7ihk'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Romans":
        file_id = '1ZoGBY-WThYG9MX_PY3imPKYsMQ7w6jSx'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "1 Corinthians":
        file_id = '1yAT1kdjk-pwUMDtreov3tcTXKo8fYIXl'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "2 Corinthians":
        file_id = '1wZcMu_PIY0yCVBmNBo46F9MHL8bmRKr7'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Galatians":
        file_id = '1sidDVgMJKLzNDrQ-dMlpIgEGWvbG6sH9'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Ephesians":
        file_id = '1_GKbBRdSqamTDxOTJBCXvR9OlvoBu5OO'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Philippians":
        file_id = '19OLUKIcWe_iY_vWErQoYmabwl4zpjzVf'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Colossians":
        file_id = '1KhdvYtOrSG-yCYhllWljbqWeZQ6eb03b'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "1 Thessalonians":
        file_id = '1-CXoIJ8UmAIyX6_9geoEWBG6ERmV0OQK'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "2 Thessalonians":
        file_id = '1acYpdLIoTQWS_NYWL1P_G-48yrg_BWiA'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "1 Timothy":
        file_id = '1akLCNTsnsTFSiNCZzgGnmDoYGdclCDPX'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "2 Timothy":
        file_id = '10Mzcj4vvO_OgovnnXPyzuo64N2J-ucwB'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Titus":
        file_id = '13byLHjosMJwYBx47Mbf0d8ldJTwwIGNa'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Philemon":
        file_id = '1FI0bEgZAUWUJlMWOOiWNORFf0ooqQJdQ'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Hebrews":
        file_id = '1_fcN7VXlwyw8BON_GAyUNmitzfxEei07'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "James":
        file_id = '1BiKc77B1DIXFlzcRZ8At8eIZSR7gmVrj'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "1 Peter":
        file_id = '1rZVqvrjyVZ7wnTJyzHQXcR7CssFkLrff'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "2 Peter":
        file_id = '1C-J3_tdztnibY6lBLtBFVnBjPCSlS5vB'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "1 John":
        file_id = '15wyoRDnKxVDcV14JJ1NHI5VuTHEAsE6Q'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "2 John":
        file_id = '195SBkMW8Xf9sMet3PZMoNEuLfWN-Xx48'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "3 John":
        file_id = '1W2H1o-MfkmDHoLiymtZqER7cxk9PSoxo/'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Jude":
        file_id = '1I-53s0pH72Sj1eEUvhGviCM7IzQbS5tm'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)
    elif book == "Revelations":
        file_id = '1AvAVZ60cMZGOm34t9sSGWkxmkl-xrkyj'
        file_url = f'https://drive.google.com/file/d/{file_id}/preview'
        show_pdf(file_url)

load_dotenv()
def main():
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.title("Select a Book")
    selected_book = st.sidebar.selectbox("Choose a Book", [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy",
        "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
        "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
        "Jude", "Revelation"
    ])

    st.header(f"Content for {selected_book}")
    show_content(selected_book)

    

    #pdf = st.file_uploader("Upload your PDF!", type='pdf')
    #show_pdf('MatthewFBV.pdf')
    #pdf = 'MatthewFBV.pdf'
    #pdf_reader = PdfReader('MatthewFBV.pdf')

    folder_path = "./Books/Bible/NewTestament"
    pdf = f"{folder_path}/{selected_book}OEV.pdf"
    print(pdf)
    pdf_reader = PdfReader(pdf)
    text = " "

    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(text=text)

    #embedding
    
    store_name = pdf[:-4]
    #st.write(store_name)
    
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        st.write('Embeddings Loaded from the Disk')
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
        st.write('Embeddings Operation Complete')

    # Input User Questions
    query = st.text_input("Ask questions about the selected book below: ")

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm = llm, chain_type = "stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = query)
            print(cb)
        st.write(response)



    #st.write(chunks)
    #st.write(text)

if __name__ == '__main__':
    main()