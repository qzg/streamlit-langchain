import tempfile
import logging
import os
import pandas as pd
from operator import itemgetter
import yaml
from yaml.loader import SafeLoader

import streamlit as st
import streamlit_authenticator as stauth

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableMap

from langchain.document_loaders import PyPDFLoader

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Define the number of docs to retrieve from the vectorstore and memory
top_k = 4

# Define the language options
lang_options = {
    "🇺🇸 English User interface":"en_US",
    "🇳🇱 Nederlandse gebruikers interface":"nl_NL"
}

#################
### Functions ###
#################

# Get authenticator and credentials
def get_authenticator():
    with open('.streamlit/credentials.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
        return stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )

# Function to load the localized strings
def localization(locale):
    # Load in the text bundle and filter by language locale
    df = pd.read_csv("localization.csv")
    df = df.query(f"locale == '{locale}'")
    # Create and return a dictionary of key/values.
    lang_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return lang_dict

# Function to load the experience on rails
def rails(username):
    # Load in the rails bundle and filter by username
    df = pd.read_csv("rails.csv")
    df = df.query(f"username == '{username}'")
    # Create and return a dictionary of key/values.
    rails_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return rails_dict

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file):
    docs = []
    if uploaded_file is not None:
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
            logging.info("""Processing: {file}""")
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

# Function to drop previously existing vector data
def drop_vector_data(username):
    session.execute(f"DROP TABLE IF EXISTS vector_preview.vector_context_{username}")

# Select the language
#with st.sidebar:
#    locale = st.selectbox(label='Language', label_visibility="hidden", options=list(lang_options.keys()))
#    lang_dict = localization(lang_options[locale])
lang_dict = localization("en_US")

######################
### Authentication ###
######################

authenticator = get_authenticator()
name, authentication_status, username = authenticator.login('Login', 'sidebar')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
elif authentication_status == False:
    with st.sidebar:
        st.error('Username/password is incorrect')
    st.cache_resource.clear()
    logging.info("Quitting for authentication")
    st.stop()
elif authentication_status == None:
    with st.sidebar:
        st.warning('Please enter your username and password')
    st.cache_resource.clear()
    logging.info("Quitting for authentication")
    st.stop()

# Select the rails experience
rails_dict = rails(username)

#####################
### Session Cache ###
#####################

# Cache Chat Memory for future runs
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        return_messages=True,
        k=top_k
    )
memory = st.session_state.memory

#######################
### Resources Cache ###
#######################

# Cache Astra DB session for future runs
with st.sidebar:
    @st.cache_resource(show_spinner=lang_dict['connect_astra'])
    def load_session():
        # Connect to Astra DB
        cluster = Cluster(cloud={'secure_connect_bundle': st.secrets["ASTRA_SCB_PATH"]}, 
                        auth_provider=PlainTextAuthProvider(st.secrets["ASTRA_CLIENT_ID"], 
                                                            st.secrets["ASTRA_CLIENT_SECRET"]))
        return cluster.connect()
    session = load_session()

# Cache OpenAI Embedding for future runs
#with st.sidebar:
#    @st.cache_resource(show_spinner=lang_dict['get_embedding'])
def load_embedding():
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
embedding = load_embedding()

# Cache Vector Store for future runs
with st.sidebar:
    @st.cache_resource(show_spinner=lang_dict['get_vectorstore'])
    def load_vectorstore(username):
        # Get the vector store from Astra DB
        return Cassandra(
            embedding=embedding,
            session=session,
            keyspace='vector_preview',
            table_name=f"vector_context_{username}"
        )
    vectorstore = load_vectorstore(username)

# Cache Retriever for future runs
#with st.sidebar:
 #   @st.cache_resource(show_spinner=lang_dict['get_retriever'])
def load_retriever():
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
retriever = load_retriever()

# Cache OpenAI Chat Model for future runs
#with st.sidebar:
#    @st.cache_resource(show_spinner=lang_dict['get_model'])
def load_model():
    # Get the OpenAI Chat Model
    return ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        streaming=True,
        verbose=True,
        )
model = load_model()

# Cache Conversational Chain for future runs
#with st.sidebar:
#    @st.cache_resource(show_spinner="Getting the Conversational Chain...")
def load_chain():

    template = """
You're a helpful AI assistent tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{history}

Question:
{question}

Answer in the user's language:
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "history": lambda x: x["history"],
        "question": lambda x: x["question"]
    }) | prompt | model

    return chain
chain = load_chain()

################
### Main app ###
################

st.title(lang_dict['title'])

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader(lang_dict['load_context'], type=['txt', 'pdf'], )
        submitted = st.form_submit_button(lang_dict['load_context_button'])
        if submitted:
            vectorize_text(uploaded_file)

# Drop the vector data and start from scratch
if username == "michel":
    with st.sidebar:
        with st.form('drop'):
            st.caption(lang_dict['drop_context'])
            submitted = st.form_submit_button(lang_dict['drop_context_button'])
            if submitted:
                with st.spinner(lang_dict['dropping_context']):
                    vectorstore.clear()
                    vectorstore = load_vectorstore(username)
                    st.session_state["messages"] = [AIMessage(content=lang_dict['assistant_welcome'])]
                    memory.clear()

# Draw rails
with st.sidebar:
        st.subheader(rails_dict[0])
        st.caption(rails_dict[1])
        for i in rails_dict:
            if i>1:
                st.markdown(f"{i-1}. {rails_dict[i]}")

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

# Redraw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
prompt = st.chat_input(lang_dict['assistant_question'])
if prompt:
     # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Draw the prompt on the page
    with st.chat_message("human"):
        st.markdown(prompt)

    # Get the results from Langchain
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        logging.info(f"Getting LLM response for: {prompt}")
        logging.info(f"Using memory: {history}")
        callback = StreamHandler(response_placeholder)
        response = chain.invoke({'question': prompt, 'history': history}, config={'callbacks':[callback]})
        logging.info(f"Response: {response}")

        # Write the final answer without the cursor
        response_placeholder.markdown(response.content)

        # Add the result to memory
        memory.save_context({'question': prompt}, {"output": response.content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=response.content))