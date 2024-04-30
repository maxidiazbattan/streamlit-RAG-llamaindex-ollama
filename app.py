# tools
import glob
from tempfile import NamedTemporaryFile

# ollama
import ollama

# llamaindex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex

# ollama
import streamlit as st


def response_generator(stream):
    """Generator that yields chunks of data from a stream response.
    Args:
        stream: The stream object from which to read data chunks.
    Yields:
        bytes: The next chunk of data from the stream response.
    """
    for chunk in stream.response_gen:
        yield chunk

@st.cache_resource(show_spinner=False)
def load_data(document, model_name:str) -> VectorStoreIndex:
    """Loads and indexes Streamlit documentation using Ollama and Llamaindex.

    This function takes a model name as input and performs the following actions:

    1. Ollama Initialization: Initializes an Ollama instance using the provided model name. Ollama is a library that facilitates communication with large language models (LLMs).
    2. Data Ingestion: Reads the Streamlit documentation (assumed to be a PDF file) using the SimpleDirectoryReader class.
    3. Text Splitting and Embedding: Splits the loaded documents into sentences using the SentenceSplitter class and generates embeddings for each sentence using the HuggingFaceEmbedding model.
    4. Service Context Creation: Creates a ServiceContext object that holds all the necessary components for processing the data, including the Ollama instance, embedding model, text splitter, and a system prompt for the LLM.
    5. VectorStore Indexing: Creates a VectorStoreIndex instance from the processed documents and the service context. VectorStore is a library for efficient searching of high-dimensional vectors.

    Args:
        # docs_path  (str): Path of the documents to query.
        model_name (str): The name of the LLM model to be used by Ollama.

    Returns:
        VectorStoreIndex: An instance of VectorStoreIndex containing the indexed documents and embeddings.
    """

    # llm
    llm = Ollama(model=model_name, request_timeout=30.0)

    # data ingestion
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(document.getbuffer())
        with st.spinner(text="Loading and indexing the Streamlit docs. This should take 1-2 minutes."):
            # loading document
            docs = SimpleDirectoryReader(".").load_data()

            # embeddings | query container
            text_splitter = SentenceSplitter(chunk_size=512)
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") # BAAI/bge-small-en-v1.5 | BAAI/bge-base-en-v1.5
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model,
                text_splitter=text_splitter,
                system_prompt="You are an Python expert and your job is to answer technical questions. Keep your answers technical and based on facts.")

            # indexing db
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

def main() -> None:
    """Controls the main chat application logic using Streamlit and Ollama.

    This function serves as the primary orchestrator of a chat application with the following tasks:

    1. Page Configuration: Sets up the Streamlit page's title, icon, layout, and sidebar using st.set_page_config.
    2. Model Selection: Manages model selection using st.selectbox and stores the chosen model in Streamlit's session state.
    3. Chat History Initialization: Initializes the chat history list in session state if it doesn't exist.
    4. Data Loading and Indexing: Calls the load_data function to create a VectorStoreIndex from the provided model name.
    5. Chat Engine Initialization: Initializes the chat engine using the VectorStoreIndex instance, enabling context-aware and streaming responses.
    6. Chat History Display: Iterates through the chat history messages and presents them using Streamlit's chat message components.
    7. User Input Handling:
          - Accepts user input through st.chat_input.
          - Appends the user's input to the chat history.
          - Displays the user's message in the chat interface.
    8. Chat Assistant Response Generation:
          - Uses the chat engine to generate a response to the user's prompt.
          - Displays the assistant's response in the chat interface, employing st.write_stream for streaming responses.
          - Appends the assistant's response to the chat history.

    Args:
        docs_path (str): Path of the documents to query.
    """

    st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with documents 💬")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    with st.sidebar:
        # model selection
        if "model" not in st.session_state:
            st.session_state["model"] = ""
        models = [model["name"] for model in ollama.list()["models"]]
        st.session_state["model"] = st.selectbox("Select a model", models)
        
        # llm
        llm = Ollama(model=st.session_state["model"], request_timeout=30.0)

        # data ingestion
        document = st.file_uploader("Upload a PDF file to query", type=['pdf'], accept_multiple_files=False)

        # file processing                
        if st.button('Process file'):
            index = load_data(document, st.session_state["model"])
            st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        # initialize chat history                   
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # index = load_data(docs_path, st.session_state["model"])
        if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)

        # display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # accept user input
        if prompt := st.chat_input("How I can help you?"):
            # add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # chat assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                message_placeholder = st.empty()
                with st.chat_message("assistant"):
                    stream = st.session_state.chat_engine.stream_chat(prompt)
                    response = st.write_stream(response_generator(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})
                
    else:
        st.markdown("<span style='font-size:15px;'><b>Upload a PDF to start chatting</span>", unsafe_allow_html=True)

if __name__=='__main__':
    main()