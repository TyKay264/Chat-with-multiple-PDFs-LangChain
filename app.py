import os
os.environ["TESSDATA_PREFIX"] = r"D:\Learning FPT\LangChain\Tesseract-OCR"
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Learning FPT\LangChain\Tesseract-OCR\tesseract.exe"

from pdf2image import convert_from_bytes
from PIL import Image

from dotenv import load_dotenv

import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from htmlTemplates import css, bot_template, user_template


# ----- Extract text from image-based PDFs using OCR -----
def get_ocr_text(pdf_docs):
    all_text = ""
    for i, pdf in enumerate(pdf_docs):
        pdf.seek(0)
        images = convert_from_bytes(pdf.read(), dpi=300)  # convert PDF to images
        file_text = ""
        for j, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="vie")  # OCR each page
            if text.strip():
                file_text += text + "\n"
            else:
                print(f"📄 File {i+1}, Page {j+1} has NO TEXT via OCR.")
        print(f"📝 OCR Extracted File {i+1} - length: {len(file_text)}")
        if(i+1 == 3):
            print(file_text)
        
        all_text += file_text
    return all_text


# ----- Split text into chunks -----
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# ----- Create vectorstore -----
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# ----- Setup LLM + memory -----
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.5,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# ----- Handle user chat -----
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        msg = message.content
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)


# ----- Main -----
def main():
    load_dotenv()
    st.set_page_config(page_title="OCR PDF Chat", page_icon="📄")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Scanned PDFs (OCR) 📄")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload scanned/image-based PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Running OCR and building knowledge base..."):
                raw_text = get_ocr_text(pdf_docs)
                if not raw_text.strip():
                    st.error("❌ No text extracted via OCR.")
                    return

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Ready! You can start chatting now.")

if __name__ == '__main__':
    main()
