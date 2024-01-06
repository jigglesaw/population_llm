import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,CSVLoader,UnstructuredCSVLoader,UnstructuredImageLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader


with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    csv_loader = DirectoryLoader('csv world/', glob="*.csv", loader_cls=UnstructuredCSVLoader)
    pdf_loader = DirectoryLoader('data/', glob =".pdf", loader_cls=PyPDFLoader)
    loaders = [csv_loader,pdf_loader]
    documents=[]
    for loader in loaders:
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

if __name__ == "__main__":
    run_db_build()