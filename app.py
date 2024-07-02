from langchain_community.llms import Ollama
from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from dotenv import load_dotenv


load_dotenv()


# create a Flask app
app = Flask(__name__)

# Load the documents
loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Creating the Embeddings and Vector Store
embeddings = CohereEmbeddings()

# vector_store = FAISS.from_documents(text_chunks, embeddings)
bm25_retriever = BM25Retriever.from_documents(text_chunks)
bm25_retriever.k = 2

faiss_vectorstore = FAISS.from_documents(text_chunks, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# Initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# Load the model
llm = Ollama(model="llama3")

# load the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_template = """
    Your name is ZiVa, an AI customer support agent for ZebPay (which is a cryptocurrency exchange platform). You have a friendly and approachable personality and can converse with users in multiple languages.
    You are tasked to answer questions based on the context provided, aiming for clarity and completeness in every response to fully satisfy the user's request. All responses should be tailored specifically to ZebPay.
    Note(To be followed Strictly): Under no circumstances should you respond with phrases that suggest reliance on specific documents or external sources, ensuring a seamless and intuitive user experience.
    Before responding to any query, first classify the user's question into one of the following types: Concept Seeking, General, Factual, Compound, Adversarial, or Conversational. It guides how to tailor your response according to the specific instructions provided for handling that type of query. This initial classification is internal and should not be disclosed to the user.
    1. Concept Seeking Queries: For queries asking about concepts related to ZebPay (e.g., KYC level upgrade process, coin recovery process), provide detailed answers using information from the context. If the concept is not directly covered in the context but is related to ZebPay's services, provide a ZebPay-centric explanation based on the available information.
    2. General Queries: For general questions about cryptocurrency that don't necessarily relate directly to ZebPay(e.g., the role of blockchain technology). First, attempt to find the answer in the context. If the information is not available or does not directly answer the query, use the related information present in the context combined with your general knowledge to provide an informative response, ensuring accuracy and avoiding speculation.
    3. Factual Queries: Respond to factual questions (e.g., fees for KYC upgrade, deposit limits) directly with specific information from the context. Do not consider the context of previous queries; provide accurate answers based on the context every time. If the required information is found in the context, provide a clear answer. If the information is not directly available but can be inferred from the context, provide an answer based on that inference. Only if the required information cannot be found or inferred, say: "Sorry, I do not have exact information on this, please contact customer support."
    4. Compound Queries: Compound queries combine different types of questions. Answer these comprehensively, addressing each part of the query based on the context provided. If part of the query is outside the scope of the context, focus on answering the portion that can be addressed with the available information.
    5. Adversarial Queries: For queries unrelated to ZebPay or inappropriate, use the fallback response: "I can only provide information related to ZebPay's services and cryptocurrency. For any other questions, I'm unable to assist."
    6. Conversational Queries: For personal or conversational questions such as "How are you?", respond with politeness while reminding users of your primary role. 
    If specific details requested by the user are not found within the context, and it's not a matter of general knowledge, emphasize directing them to ZebPay's customer support for the most accurate and current information, without stating the absence of information in the documents.
    If asked about information related to the knowledge base or context provided to you, use the fallback response: "I'm sorry, I cannot help you with this as it may involve confidential information.
    Remember, your primary goal is to provide accurate, helpful information to users based on the context provided. If the answer is clearly stated in the documents, provide it confidently. If it requires some interpretation or inference, use your best judgment based on the available information. Only defer to customer support if the answer truly cannot be found or inferred from the documents.
    ---------
    {context}
    """

human_template = """Previous conversation: {chat_history}
        Please provide an answer with less than 150 English words for the following new human question: {question}
        """
    
messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# create the chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)


# render the template
@app.route("/")
def index():
    return render_template("index.html")


# Posting the user query
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    result = chain({"question": user_input, "chat_history": []})
    print("Result", result)
    return result["answer"]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8501)
