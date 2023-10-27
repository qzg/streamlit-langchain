import streamlit as st

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

st.title('🦜🔗 Enterprise Chat Agent')

# Cache Astra DB session for future runs
@st.cache_resource(show_spinner="Setting up Astra DB connection...")
def load_session():
    # Connect to Astra DB
    cluster = Cluster(cloud={'secure_connect_bundle': st.secrets["ASTRA_SCB_PATH"]}, 
                      auth_provider=PlainTextAuthProvider(st.secrets["ASTRA_CLIENT_ID"], 
                                                          st.secrets["ASTRA_CLIENT_SECRET"]))
    return cluster.connect()

session = load_session()

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner="Getting the OpenAI embedding...")
def load_embedding():
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

embedding = load_embedding()

# Cache Vector Store for future runs
@st.cache_resource(show_spinner="Getting the Vector Store from Astra DB...")
def load_vectorstore():
    # Get the vector store from Astra DB
    return Cassandra(
        embedding=embedding,
        session=session,
        keyspace='vector_preview',
        table_name='romeo'
    )

vectorstore = load_vectorstore()

# Cache OpenAI Chat Model for future runs
@st.cache_resource(show_spinner="Getting the OpenAI Chat Model...")
def load_model():
    # Get the OpenAI Chat Model
    return ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        verbose=True)

model = load_model()

# Prepare the system message
message_system = SystemMessage(content="You're are a helpful, " 
                                        "talkative, and friendly assistant.")

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Now get a prompt from a user
if prompt := st.chat_input("What is up?"):
    # Add the prompt to messages, stored in session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Draw the prompt on the page
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the user message using the value from the prompt
    message_user = HumanMessage(content=prompt)

    # Get the results from Langchain
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        message_placeholder = st.empty()

        # Stream the results in
        full_response = ""
        for response in model.stream([message_system, message_user]):
            wordstream = response.dict().get('content')

             # if wordstream is not None
            if wordstream:
                full_response += wordstream
                # This message_placeholder is a st.empty from the display
                message_placeholder.markdown(full_response + "▌")

        # Overwrite the final answer to get rid of the cursor
        message_placeholder.markdown(full_response)

    # Add the answer to the messages session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})