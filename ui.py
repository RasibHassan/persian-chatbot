import streamlit as st
import os
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
from pinecone_text.sparse import BM25Encoder
import os

import nltk
nltk.download('punkt_tab')

# API keys remain the same
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def debug_print_context(inputs):
    """Debug function to print context details."""
    con = inputs.get("context", [])
    context = []
    for doc in con:
        context.append(doc.metadata)
    return inputs

def create_chatbot_retrieval_qa(main_query, additional_note, vs, categories, sub_categories, top_k):
    """Modified to handle both main query and additional note with custom top_k value."""
    prompt_template = """
    شما یک دستیار هوشمند و مفید هستید. با استفاده از متن زیر به پرسش مطرح‌شده با دقت، شفافیت، و به صورت کامل پاسخ دهید:
    1. پاسخ را **به زبان فارسی** ارائه دهید.
    2. **جزئیات کامل** را پوشش دهید و اطمینان حاصل کنید که تمام جنبه‌های سؤال به دقت بررسی شده‌اند.
    3. تاریخ‌ها و اطلاعات ارائه‌شده باید **مطابق با متن** باشند. از درج تاریخ‌های نادرست خودداری کنید.
    4. در صورت نیاز به ارجاع به تاریخ، از **نام فایل برای تاریخ دقیق** استفاده کنید.
    5. نام فایل را در مرجع پاسخ بدهید

    **متن:**
    {context}

    **سؤال اصلی:**
    {main_question}

    **یادداشت اضافی:**
    {additional_note}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)

    def filtered_retriever(query):
        filter_dict = {}
        if categories:
            filter_dict["category"] = {"$in": ['اطلاعیه ـ از 1995']}
        if sub_categories != ['ALL'] and sub_categories != []:
            if sub_categories:
                filter_dict["year"] = {"$in": sub_categories}
        
        return vs.get_relevant_documents(
            query,
            filter=filter_dict,
            top_k=top_k  # Use the custom top_k value
        )

    chain = (
        {
            "context": lambda x: filtered_retriever(x["main_question"]),
            "main_question": lambda x: x["main_question"],
            "additional_note": lambda x: x["additional_note"]
        }
        | RunnablePassthrough(lambda inputs: debug_print_context(inputs))
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain

def initialize_chatbot(top_k=40):
    """Initialize the chatbot with Pinecone index and embeddings with custom top_k."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "persian-new"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    index = pc.Index(INDEX_NAME)

    bm25_encoder = BM25Encoder().load("full_bm25_values.json")

    vectorstore = PineconeHybridSearchRetriever(
        alpha=0.3, 
        embeddings=embeddings, 
        sparse_encoder=bm25_encoder, 
        index=index,
        top_k=top_k  # Use the custom top_k value
    )

    return vectorstore

def main():
    st.markdown("<h1 class='persian-text'>چت‌بات فارسی</h1>", unsafe_allow_html=True)

    # Initialize session state for loading
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Add search depth controls
    st.markdown("### عمق جستجو")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        simple_search = st.checkbox("جستجوی ساده")
    with col2:
        complex_search = st.checkbox("جستجوی پیچیده")
    with col3:
        deep_search = st.checkbox("جستجوی عمیق")

    # Determine top_k based on selected search depth
    if simple_search:
        top_k = 25
    elif complex_search:
        top_k = 60
    elif deep_search:
        top_k = 100
    else:
        top_k = 40  # Default value

    # Rest of the settings
    sub_cat = ['ALL','1995_NCRI', '1996_NCRI', '1997_NCRI', '1998_NCRI', '1999_NCRI', '2001_NCRI', '2002_NCRI', '2003_NCRI', '2004_NCRI', '2005_NCRI', '2006_NCRI', '2007_NCRI', '2008_NCRI', '2009_NCRI', '2010_NCRI', '2011_NCRI', '2012_NCRI', '2013_NCRI', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'PMOI', 'دفاتر كشورها قبل از 2003', 'كميسيونها قبل از 2003']
    cat = ['اطلاعیه ـ از 1995']

    # Category selections
    categories = st.multiselect(
        "دسته‌بندی را انتخاب کنید:",
        cat,
        default=[]
    )
    sub_categories = st.multiselect(
        "زیر دسته‌بندی را انتخاب کنید:",
        sub_cat,
        default=[]
    )

    # Input fields
    main_query = st.text_area(
        "سؤال اصلی خود را اینجا وارد کنید:",
        height=100
    )

    additional_note = st.text_area(
        "یادداشت اضافی (اختیاری):",
        height=100
    )

    # Initialize chatbot with current top_k value
    if 'vectorstore' not in st.session_state or 'current_top_k' not in st.session_state or st.session_state.current_top_k != top_k:
        with st.spinner('در حال راه‌اندازی چت‌بات...'):
            try:
                st.session_state.vectorstore = initialize_chatbot(top_k)
                st.session_state.current_top_k = top_k
            except Exception as e:
                st.error(f"خطا در راه‌اندازی chatbot: {e}")
                return

    # Create placeholder for response
    response_placeholder = st.empty()

    # Submit button
    if st.button("ارسال"):
        if not main_query:
            st.warning("لطفاً سؤال اصلی خود را وارد کنید.")
            return

        if not categories and not sub_categories:
            st.warning("لطفاً حداقل یک دسته‌بندی یا زیر دسته‌بندی را انتخاب کنید.")
            return

        try:
            with st.spinner('لطفاً صبر کنید...'):
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("در حال جستجوی اطلاعات مرتبط...")
                progress_bar.progress(33)
                
                chatbot = create_chatbot_retrieval_qa(
                    main_query,
                    additional_note,
                    st.session_state.vectorstore,
                    categories,
                    sub_categories,
                    top_k
                )
                
                status_text.text("در حال پردازش اطلاعات...")
                progress_bar.progress(66)
                
                response = chatbot.invoke({
                    "main_question": main_query,
                    "additional_note": additional_note if additional_note else ""
                })
                
                status_text.text("در حال آماده‌سازی پاسخ...")
                progress_bar.progress(100)
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                response_placeholder.markdown("**پاسخ:**")
                response_placeholder.write(response)

        except Exception as e:
            st.error(f"خطا در پردازش سوال: {e}")
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()