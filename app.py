import os
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import sqlite3

from langchain.chat_models import ChatOpenAI

from langchain_groq import ChatGroq

from langchain_community.chat_models import ChatOllama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from htmlTemplates import css, bot_template, user_template

from db import create_tables, insert_document, insert_chunks, get_chunks_by_file, get_all_documents

from langchain.prompts import PromptTemplate

# ----- Extract text from a single PDF -----
def get_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf_reader:
        for page_index, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                text += content + "\n"
            else:
                print(f"📄 File {pdf_file.name}, Page {page_index + 1} has NO TEXT.")
    print(f"📝 {pdf_file.name} - length: {len(text)}")
    return text


# ----- Split text into chunks -----
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# ----- Create new FAISS index from documents -----
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# ----- Load FAISS from disk -----
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)



# ----- Build conversation chain -----
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo",  # hoặc "gpt-3.5-turbo"
    #     temperature=0.5,
    #     openai_api_key=os.getenv("OPENAI_API_KEY")
    # )

    # llm = ChatGroq(
    #     # model_name="llama3-70b-8192",
    #     model_name="mixtral-8x7b-32768",
    #     temperature=0.5,
    #     groq_api_key=os.getenv("GROQ_API_KEY")
    # )

    # llm = ChatOllama(
    #     model="gemma3",
    #     # temperature=0.5
    # )

    # llm = ChatOllama(
    #     model="qwen3",
    #     temperature=0.5
    # )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

    custom_prompt = PromptTemplate.from_template("""
    Bạn là trợ lý AI, chỉ trả lời bằng tiếng Việt dựa trên thông tin từ các tài liệu sau:

    {context}   

    Câu hỏi: {question}
    ❗ Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời: "Tôi không biết dựa trên tài liệu đã cung cấp."
    """)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

# ----- Handle user input -----
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# ----- Main App -----
def main():
    load_dotenv()
    create_tables()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📄")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # ✅ Tự load FAISS nếu đã từng xử lý
    if st.session_state.conversation is None and os.path.exists("faiss_index/index.faiss"):
        vectorstore = load_vectorstore()
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.info("📂 Đã nạp dữ liệu từ lần xử lý trước. Bạn có thể đặt câu hỏi ngay.")

    # ----- UI chính -----
    st.header("Chat with multiple PDFs 📄")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        # ✅ Hiển thị danh sách tài liệu đã xử lý
        st.markdown("### 📁 Tài liệu đã xử lý trước:")
        for name in get_all_documents():
            st.markdown(f"- {name}")

        if st.button("Process") and pdf_docs:
            with st.spinner("🔄 Processing..."):
                all_chunks = []

                for pdf in pdf_docs:
                    file_name = pdf.name
                    text = get_pdf_text(pdf)
                    if not text.strip():
                        continue

                    document_id = insert_document(file_name)
                    existing_chunks = get_chunks_by_file(file_name)

                    if existing_chunks:
                        st.info(f"✅ Đã tồn tại dữ liệu cho {file_name}. Sử dụng lại.")
                        chunks = existing_chunks
                    else:
                        chunks = get_text_chunks(text)
                        insert_chunks(document_id, chunks)

                    all_chunks.extend([
                        Document(page_content=f"[SOURCE: {file_name}]\n{chunk}", metadata={"source": file_name})
                        for chunk in chunks
                    ])

                if not all_chunks:
                    st.error("❌ Không có nội dung nào được xử lý.")
                    return

                vectorstore = get_vectorstore(all_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Xử lý hoàn tất! Bạn có thể đặt câu hỏi.")

if __name__ == '__main__':
    main()
