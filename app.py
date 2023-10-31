import logging
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import CassandraChatMessageHistory

import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableMap
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

from typing import Any, List, Dict

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=''):
        self.container = container
        self.text = initial_text

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.text = ''

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + '▌')

print("Started")

#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 4
top_k_memory = 3

# Define the language options
lang_options = {
    '🇺🇸 English User interface':'en_US',
    '🇳🇱 Nederlandse gebruikers interface':'nl_NL'
}

###############
### Globals ###
###############

global lang_dict
global rails_dict
global session
global embedding
global vectorstore
global retriever
global model
global chat_history
global memory
global authenticator

#################
### Functions ###
#################

# Get authenticator and credentials
def load_authenticator():
    print("load_authenticator")
    with open('.streamlit/credentials.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
        return stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file):
    if uploaded_file is not None:
        docs = []

        if uploaded_file.name.endswith('txt'):
            file = [uploaded_file.read().decode()]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 200
            )  
            texts = text_splitter.create_documents(file)
            vectorstore.add_documents(texts)  
            st.info(f"{len(texts)} {lang_dict['load_text']}")

        if uploaded_file.name.endswith('pdf'):
            
            # Read PDF
            docs = []
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print("""Processing: {file}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 200
            )  
            pages = text_splitter.split_documents(docs)
            vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} {lang_dict['load_pdf']}")

################################
### Resources and Data Cache ###
################################

# Cache localized strings
@st.cache_data()
def load_localization(locale):
    print("load_localization")
    # Load in the text bundle and filter by language locale
    df = pd.read_csv("localization.csv")
    df = df.query(f"locale == '{locale}'")
    # Create and return a dictionary of key/values.
    lang_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return lang_dict
lang_dict = load_localization('en_US')

# Cache localized strings
@st.cache_data()
def load_rails(username):
    print("load_rails")
    # Load in the rails bundle and filter by username
    df = pd.read_csv("rails.csv")
    df = df.query(f"username == '{username}'")
    # Create and return a dictionary of key/values.
    rails_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return rails_dict

# Cache Astra DB session for future runs
@st.cache_resource(show_spinner=lang_dict['connect_astra'])
def load_session():
    print("load_session")
    # Connect to Astra DB
    cluster = Cluster(cloud={'secure_connect_bundle': st.secrets["ASTRA_SCB_PATH"]}, 
                    auth_provider=PlainTextAuthProvider(st.secrets["ASTRA_CLIENT_ID"], 
                                                        st.secrets["ASTRA_CLIENT_SECRET"]))
    return cluster.connect()

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner=lang_dict['load_embedding'])
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

# Cache Vector Store for future runs
@st.cache_resource(show_spinner=lang_dict['load_vectorstore'])
def load_vectorstore(username):
    print("load_vectorstore")
    # Get the load_vectorstore store from Astra DB
    return Cassandra(
        embedding=embedding,
        session=session,
        keyspace='vector_preview',
        table_name=f"vector_context_{username}"
    )
    
# Cache Retriever for future runs
@st.cache_resource(show_spinner=lang_dict['load_retriever'])
def load_retriever():
    print("load_retriever")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

# Cache OpenAI Chat Model for future runs
@st.cache_resource(show_spinner=lang_dict['load_model'])
def load_model():
    print("load_model")
    # Get the OpenAI Chat Model
    return ChatOpenAI(
        model='gpt-3.5-turbo-16k',
        openai_api_key=st.secrets['OPENAI_API_KEY'],
        streaming=True,
        verbose=False
    )

# Cache Chat History for future runs
@st.cache_resource(show_spinner=lang_dict['load_message_history'])
def load_chat_history(username):
    print("load_chat_history")
    return CassandraChatMessageHistory(
        session_id=username,
        session=session,
        keyspace='vector_preview',
        ttl_seconds = 864000 # Ten days
    )

@st.cache_resource(show_spinner=lang_dict['load_message_history'])
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

# Cache prompt
@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You're a helpful AI assistent tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in the user's language:"""

    return ChatPromptTemplate.from_template(template)

#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

############
### Main ###
############

authenticator = load_authenticator()
name, authentication_status, username = authenticator.login('Login', 'sidebar')

if authentication_status != None:
    authenticator.logout('Logout', 'sidebar')
    
    with st.sidebar:
        rails_dict = load_rails(username)
        session = load_session()
        embedding = load_embedding()
        vectorstore = load_vectorstore(username)
        retriever = load_retriever()
        model = load_model()
        chat_history = load_chat_history(username)
        memory = load_memory()
        prompt = load_prompt()

        # Include the upload form for new data to be Vectorized
    with st.sidebar:
        with st.form('upload'):
            uploaded_file = st.file_uploader(lang_dict['load_context'], type=['txt', 'pdf'], )
            submitted = st.form_submit_button(lang_dict['load_context_button'])
            if submitted:
                vectorize_text(uploaded_file, st.session_state.vectorstore)

    # Drop the vector data and start from scratch
    if username == 'michel':
        with st.sidebar:
            with st.form('drop'):
                st.caption(lang_dict['drop_context'])
                submitted = st.form_submit_button(lang_dict['drop_context_button'])
                if submitted:
                    with st.spinner(lang_dict['dropping_context']):
                        st.session_state.vectorstore.clear()
                        st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]
                        st.session_state.memory.clear()

    # Draw rails
    with st.sidebar:
            st.subheader(rails_dict[0])
            st.caption(rails_dict[1])
            for i in rails_dict:
                if i>1:
                    st.markdown(f"{i-1}. {rails_dict[i]}")

    # Draw all messages, both user and agent so far (every time the app reruns)
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)

    # Now get a prompt from a user
    if question := st.chat_input(lang_dict['assistant_question']):
        print(f"Got question {question}")

        # Add the prompt to messages, stored in session state
        st.session_state.messages.append(HumanMessage(content=question))

        # Draw the prompt on the page
        with st.chat_message('human'):
            st.markdown(question)

        # Get the results from Langchain
        with st.chat_message('assistant'):
            # UI placeholder to start filling with agent response
            response_placeholder = st.empty()
            callback = StreamHandler(response_placeholder)

            history = memory.load_memory_variables({})
            print(f"Using memory: {history}")

            chain = RetrievalQA.from_chain_type(
                llm=model,
                retriever=retriever,
                return_source_documents=False,
                verbose=False,
                chain_type_kwargs={"prompt": prompt}
            )

            chain = RunnableMap({
                'context': lambda x: retriever.get_relevant_documents(x['question']),
                'chat_history': lambda x: x['chat_history'],
                'question': lambda x: x['question']
            }) | prompt | model

            #response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks':[callback]})
            full_response = ""
            for chunk in chain.stream({'question': question, 'chat_history': history}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

            print(f"Response: {full_response}")

            # Write the final answer without the cursor
            response_placeholder.markdown(full_response)

            # Add the result to memory
            memory.save_context({'question': question}, {'answer': full_response})

            # Add the answer to the messages session state
            st.session_state.messages.append(AIMessage(content=full_response))

    with st.sidebar:
                st.caption("v3110_04")

elif authentication_status == False:
    with st.sidebar:
        st.error('Username/password is incorrect')
    print('Username/password is incorrect')

elif authentication_status == None:
    with st.sidebar:
        st.warning('Please enter your username and password')
    print('Please enter your username and password')