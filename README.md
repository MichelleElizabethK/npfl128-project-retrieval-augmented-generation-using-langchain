### NPFL128 Language Technologies In Practice - Summer 2023

# Retrieval Augmented Generation | Question Answering Over Custom Data Using Langchain

Langchain is an open source framework which is used to develop applications powered by language models. This project leverages the capabilities of langchain to perform retrieval augmented generation (RAG) which involves an LLM retrieving contextual documents from an external dataset as part of its execution. Specifically, we have built a question answering system that can chat with your data.

## System Description

![System Description](system_image.jpeg)

This QA system has 5 main steps:

1. Document Loading

    This step involves loading the external source of data. Langchain provides various document loaders which extracts the data into documents.
    In particular, we support 3 formats of data - web URLs, PDF documents and YouTube videos.

2. Splitting

    Splitting refers to splitting long texts in the documents into semantically related chunks. This is done using the RecursiveCharacterTextSplitter.

3. Storage

    The third step involves creating embeddings of the split documents and indexing and storing them in a vector database. Langchain provides many options. Here we use OpenAIEmbeddings and Chroma as the embedding function and vector database respectively.

4. Retrieval

    Langchain provides many methods to retrieve the required information from the vector db which includes similarity search, contextual compression, maximum marginal relevance search, TF-IDF and SVM to name a few. These methods can also be combined to improve the search.
    The Retriever interface which exposes the index is the important part in this step.

5. Output

    The retriever is then used by a question answering chain. For simple retrieval, we use the RetrievalQA chain which does not keep track of the history. To have a full fledged chat about the documents, we use ConverstationalRetrievalChain for which we can define a memory which keeps track of the chat history as well.
    Prompts for these chains have been customised so that the bot only answers questions based on the documents and not any pretrained data.

## Getting Started

Since we are using OpenAIs models, we first need to generate an access token in your openai account. Then store this in a .env file as ```OPENAI_API_KEY="your-key"```

### Setting up the environment

Set up a virtual environment in python using the command:

``` python -m venv venv```

Activate the virtual environment using:

```source venv/bin/activate```

Now, install all the dependencies provided in the requirements.txt file by running the command:
```pip install -r requirements.txt```

### Running the application

The application is run through the command line and uses command line arguments to pass the documents/urls/videos that you want to chat with.

To run the program:

```python main.py --type=url --source=<url_of_the_webpage>```

```python main.py --type=youtube --source=<id_of_the_youtube_video>```

eg: If the link to the YouTube video is https://www.youtube.com/watch?v=Y_O-x-itHaU, then the part after '=' is the id which is Y_O-x-itHaU in this case.

```python main.py --type=pdf --source=<path_to_the_pdf_document>```


