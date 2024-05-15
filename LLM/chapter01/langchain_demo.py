import yaml
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

yaml_file = "../../key/key.yaml"
with open(yaml_file, 'r') as file:
    data_key = yaml.safe_load(file)
openai_info = data_key.get('openai-proxy', {})
openai_api_key = openai_info.get('OPENAI_API_KEY')
base_url = openai_info.get('BASE_URL')

loader = TextLoader("./data/demo2.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)

texts = text_splitter.split_documents(docs)
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key, openai_api_base=base_url
)

db = FAISS.from_documents(texts, embeddings_model)
retriever = db.as_retriever()

model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, openai_api_base=base_url)
memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history', output_key='answer')
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory
)

question = "卢浮宫这个名字怎么来的？"
qa.invoke({"chat_history": memory, "question": question})

question = "对应的拉丁语是什么呢？"
qa.invoke({"chat_history": memory, "question": question})

qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

question = "卢浮宫这个名字怎么来的？"
qa.invoke({"chat_history": memory, "question": question})
