# RAG with Llamaindex & Ollama

This repo contains an implementation of an Retrieval-Augmented Generation (RAG) chatbot leveraging LlamaIndex, Ollama and Streamlit. The chatbot efficiently retrieves relevant information from documents and generates coherent, contextually appropriate responses.
You can test this app in lightning studios via this [link](https://lightning.ai/maxidiazbattan/studios/rag-streamlit-llamaindex-ollama), or locally following the steps below.

### 1. [Install](https://github.com/ollama/ollama?tab=readme-ov-file) ollama and pull models

Start Ollama

```shell
ollama serve
```

Pull the LLM you'd like to use:

```shell
ollama pull llama3
```

### 2. Create a virtual environment

```shell
python -m venv venv
source venv/bin/activate
```

### 3. Install libraries

```shell
pip install -r requirements.txt
```

### 4. Run RAG App

```shell
streamlit run app.py
```

- Open [localhost:8501](http://localhost:8501) to view your local RAG app.

- Add PDFs and ask questions. For this case I use this paper about the inadequacy of shap values
- Example PDF: https://arxiv.org/pdf/2302.08160

