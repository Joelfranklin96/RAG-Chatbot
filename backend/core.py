from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from typing import Optional, List, Dict, Any

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm(query: str, chat_history: Optional[List[Dict[str, Any]]] = None):
    # Safer pattern for default mutable args
    if chat_history is None:
        chat_history = []
    
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)
    
    # 1. Create and execute the rephrase chain
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    rephrase_chain = rephrase_prompt | chat | StrOutputParser()
    rephrased_query = rephrase_chain.invoke({"input": query, "chat_history": chat_history})
    
    # 2. Retrieve documents with the rephrased query string
    retriever = docsearch.as_retriever()
    docs = retriever.invoke(rephrased_query)
    retrieved_context = format_docs(docs)
    
    # 3. Build final RAG answer
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rag_chain = retrieval_qa_chat_prompt | chat | StrOutputParser()
    
    # 4. Invoke with all components
    result = rag_chain.invoke({
        "input": query,  # Use original query for final answer
        "context": retrieved_context,
        "chat_history": chat_history
    })
    
    return {"answer": result, "context": docs}
