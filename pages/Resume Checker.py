#Main File
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
from design import toggle

import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter


st.set_page_config (
    page_title="Resumer Master", #name of webpage 
      page_icon="📖"
      )
st.title ("Resume Building 101")
toggle()

with st.sidebar:
    st.title("Resume Reviewer")
    st.markdown('''
    ## About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

 
    ''')
    st.write('Made by Spanish Indian Inquision')



def read_resume(jd, resume, options):
    new_resume = analyze_str(resume, options)
    resume_string = new_resume.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x).to_string(index=False)
    st.write("OpenAI comprehensive analysis..")
    summary_question = f"Job requirements: {{{jd}}}" + f"Resume summary: {{{resume_string}}}" + ", please directly provide the candidate's match summary for this position (within 200 words);'"
    summary = ask_openAI(summary_question)
    new_resume.loc[len(new_resume)] = ['Comprehensive summary', summary]
    extra_info = "Scoring requirements: top 10 universities in China +3 points, 985 universities +2 points, 211 universities +1 point, top company experience +2 points, well-known company +1 point, overseas background +3 points, foreign company background +1 point. "
    score_question = f"Job requirements: {{{jd}}}" + f"Resume summary: {{{new_resume.to_string(index=False)}}}" + ", please directly return the candidate's match score for this job (0-100), please score accurately for easy comparison and ranking for other candidates, '" + extra_info
    score = ask_openAI(score_question)
    new_resume.loc[len(new_resume)] = ['Match score', score]

    return new_resume

def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(resume)

    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    resume_data = [{'option': option, 'value': []} for option in options]
    st.write("Information Retrieval")

    # Create progress bar and empty element
    progress_bar = st.progress(0)
    option_status = st.empty()

    for i, option in tqdm(enumerate(options), desc="Information Retrieval in progress", unit="option", ncols=100):
        question = f"What is the {option} of this candidate, please return a concise answer, up to 250 words, if not found, return 'Not provided'"
        docs = knowledge_base.similarity_search(question)
        llm = OpenAI(openai_api_key=openai.api_key, temperature=0.3, model_name="text-davinci-003", max_tokens="2000")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question )
        resume_data[i]['value'] = response
        option_status.text(f"Fetching information for: {option}")

        # Update progress bar
        progress = (i + 1) / len(options)
        progress_bar.progress(progress)

    df = pd.DataFrame(resume_data)
    st.success("Resume elements retrieved")
    return df

def ask_openAI(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].text.strip()

load_dotenv()
def main():
    #openai.api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pdf = st.file_uploader("Upload a file", type='pdf')
    


    #st.write(pdf) #used for printing file name
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
            )
        
        chunks = text_splitter.split_text(text=text)

        #st.write(chunks)
        # # embeddings
        store_name = pdf.name[:-4]
        # st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

    

    
if __name__ == "__main__":
    main()



