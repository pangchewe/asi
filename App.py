import streamlit as st
import attxt  # Importing the functions from the main.py file
from langchain.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders import TextLoader
import os
import tkinter as tk
from tkinter import filedialog
import shutil
from langchain.document_loaders import TextLoader
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.document_loaders import UnstructuredPDFLoader  # load pdf
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator  # vectorize db index with chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI, google_palm
from llm2 import llm_kno_run
#llm1
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDUnyNoM6LzupQRCYpeQg5aXdGumekVbsE'

st.title("Advanced Answerx")
pdf_folder_path = './File_input/pdfs'
text_folder_path = './File_input/txts'
if (len(os.listdir(pdf_folder_path))>0 or len(os.listdir(text_folder_path))>0):
    option = st.sidebar.selectbox('Select the Model',('GeminiPro', 'DOCQNA_Model_1.0.0'))

else:
    option = st.sidebar.selectbox('Select the Model',('GeminiPro',))


uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
        # Determine destination folder based on file type
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            destination_folder = "File_input/pdfs"
        elif file_extension == "txt":
            destination_folder = "File_input/txts"
        else:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")

        # Check if destination folder exists, create if not
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Save the uploaded file to the destination folder
        file_path = os.path.join(destination_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File uploaded and saved successfully to {file_path}")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": " 🤖 Hey did you just cleard our convo :( "}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Button for recording audio
        # Display a file uploader widget
pdf_folder_path = './File_input/pdfs'
text_folder_path = './File_input/txts'
if (len(os.listdir(pdf_folder_path))>0 or len(os.listdir(text_folder_path))>0) and option=="DOCQNA_Model_1.0.0":

    def base():
        pdf_folder_path = './File_input/pdfs'
        text_folder_path = './File_input/txts'

        if len(os.listdir(pdf_folder_path))>0 and len(os.listdir(text_folder_path))>0:
            pdf_loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
            text_loaders = [TextLoader(os.path.join(text_folder_path, fn)) for fn in os.listdir(text_folder_path)]
            combined_loaders = text_loaders + pdf_loaders

        if len(os.listdir(text_folder_path))>0 and len(os.listdir(pdf_folder_path))==0:
            text_loaders = [TextLoader(os.path.join(text_folder_path, fn)) for fn in os.listdir(text_folder_path)]
            combined_loaders = text_loaders
        
        if len(os.listdir(text_folder_path))==0 and len(os.listdir(pdf_folder_path))>0:
            pdf_loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
            combined_loaders = pdf_loaders

        combined_index = VectorstoreIndexCreator(
            embedding=GooglePalmEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(combined_loaders)
        
        return combined_index

    combined_index = base()
    print("Vector index created")

    pdf_chain = RetrievalQA.from_chain_type(llm=GooglePalm(),
                                        retriever=combined_index.vectorstore.as_retriever(),
                                        input_key="question",
                                        chain_type="stuff")


    def llm1_doc_run(prompt):
        pdf_answer = pdf_chain.run(prompt)
        return pdf_answer

    text = None

    if promt := st.chat_input():
                # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": promt})
            # Display user message in chat message container
            with st.chat_message("user"):
                    st.markdown(promt)
                    llm1_response = llm1_doc_run(promt)
                    



    if st.button("Record Audio"):
        text = attxt.button()
        st.session_state.messages.append({"role": "user", "content": text})
        # Display user message in chat message container
        with st.chat_message("user"):
                st.markdown(text)
                llm1_response = llm1_doc_run(text)
    
    if promt or text:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = llm1_response
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response:
                full_response += chunk + ""
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
                # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if (len(os.listdir(pdf_folder_path))==0 and len(os.listdir(text_folder_path))==0) or  option=="GeminiPro":
        text = None

        if promt := st.chat_input():
                    # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": promt})
                # Display user message in chat message container
                with st.chat_message("user"):
                        st.markdown(promt)
                        llm2_response = llm_kno_run(promt)
            
        if st.button("Record Audio"):
                text = attxt.button()
                st.session_state.messages.append({"role": "user", "content": text})
                # Display user message in chat message container
                with st.chat_message("user"):
                        st.markdown(text)
                        llm2_response = llm_kno_run(text)
     
        if promt or text:
         with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = llm2_response
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response:
                full_response += chunk + ""
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
                # Add assistant response to chat history
         st.session_state.messages.append({"role": "assistant", "content": full_response})