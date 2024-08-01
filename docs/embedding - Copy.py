from langchain_community.document_loaders import WebBaseLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_chroma import Chroma
import warnings
import os

import time
embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )

path = "./Vectorstore_folder_BKI_tests"

if os.path.exists(path):
    print('found')
    vector_db = Chroma(
        persist_directory = path,
        embedding_function = embedding
    )

else:

    # #sitespy
    # print('not found')
    # loader = WebBaseLoader(
    #     web_path=["https://barata.ai/about/",'https://barata.ai/','https://barata.ai/service/','https://barata.ai/product/','https://barata.ai/contact/']
    # )

    # docs = loader.load()

    # #md
    # markdown_path = "employees.md"
    # loader = UnstructuredMarkdownLoader(markdown_path)
    # additional_docs = loader.load()
    # all_docs = docs + additional_docs
    # pdf
    pdf_files = "publication_bki.pdf"
    loader = PyPDFLoader(pdf_files)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    import tiktoken
    all_splits = text_splitter.split_documents(docs)
    total_token_count = sum(len(tiktoken.encoding_for_model("gpt-4o-mini").encode(doc.page_content)) for doc in all_splits)
    print(total_token_count)
    # vector_db = Chroma.from_documents(
    #     all_splits,
    #     embedding,
    #     persist_directory=path
    #     )
    # print('Vector store creation is finish')


retriever = vector_db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2}
)
# query = 'Class designation for hull'
# retrieved_docs = retriever.invoke(query)
# retrieved_doc = retrieved_docs[0]
# print(vector_db)
# print(retrieved_docs)
# print(retrieved_doc.metadata)
# print(retrieved_doc.metadata['page'])
