import streamlit as st
import os
import tempfile
import json
from collections import Counter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI 
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

LOCALES = {
    "English": {
        "title": "GIS Document Assistant",
        "config": "⚙️ Configuration",
        "lang_label": "🌐 UI Language / لغة الواجهة",
        "provider": "LLM Provider",
        "model": "Model",
        "api_lbl": "Enter API Key",
        "upload_hd": "📄 Upload Documents",
        "upload_lbl": "Upload GIS PDFs",
        "process_btn": "Process Documents",
        "stats_hd": "📊 Usage Statistics",
        "total_q": "Total Questions Asked",
        "top_p": "Top Used Pages",
        "download_btn": "💾 Download Chat as JSON",
        "ask_placeholder": "Ask a question about your GIS documents...",
        "thinking": "Analyzing documents...",
        "sources_hd": "📚 View Document Sources",
        "step_1_msg": "👈 Step 1: Please configure your LLM Provider and enter your API Key in the sidebar.",
        "step_2_msg": "👈 Step 2: API Key received! Now please upload your GIS PDF documents in the sidebar and click 'Process Documents'.",
        "unlock_upload": "🔑 Enter your API key above to unlock file uploads.",
        "file_warn": "Please upload at least one PDF file.",
        "success_msg": "Successfully processed {} document(s) into {} chunks!",
        "suggested_hd": "💡 Suggested Questions"
    },
    "العربية": {
        "title": "مساعد مستندات نظم المعلومات الجغرافية (GIS)",
        "config": "⚙️ الإعدادات",
        "lang_label": "🌐 UI Language / لغة الواجهة",
        "provider": "مزود خدمة الـ LLM",
        "model": "النموذج (Model)",
        "api_lbl": "أدخل مفتاح الـ API Key",
        "upload_hd": "📄 رفع المستندات",
        "upload_lbl": "ارفع ملفات نظم المعلومات الجغرافية PDF",
        "process_btn": "معالجة المستندات",
        "stats_hd": "📊 إحصائيات الاستخدام",
        "total_q": "إجمالي الأسئلة المطروحة",
        "top_p": "الصفحات الأكثر استخداماً",
        "download_btn": "💾 تحميل المحادثة كملف JSON",
        "ask_placeholder": "اسأل سؤالاً حول مستندات الـ GIS الخاصة بك...",
        "thinking": "جاري تحليل المستندات...",
        "sources_hd": "📚 عرض مصادر المستندات",
        "step_1_msg": "👈 الخطوة 1: يرجى اختيار مزود الـ LLM وإدخال مفتاح الـ API في القائمة الجانبية للبدء.",
        "step_2_msg": "👈 الخطوة 2: تم إدخال مفتاح الـ API بنجاح! الآن يرجى رفع ملفات الـ PDF والضغط على 'معالجة المستندات'.",
        "unlock_upload": "🔑 أدخل مفتاح الـ API بالأعلى لفتح ميزة رفع المستندات.",
        "file_warn": "يرجى رفع ملف PDF واحد على الأقل.",
        "success_msg": "تمت معالجة {} ملف(ات) بنجاح إلى {} جزءاً!",
        "suggested_hd": "💡 أسئلة مقترحة"
    }
}

st.set_page_config(page_title="ITI GIS Assistant", page_icon="🗺️", layout="wide")

st.markdown("""
    <style>
    h1, h2, h3 { color: #E20613 !important; }
    .stButton>button { background-color: #E20613; color: white; border-radius: 5px; border: none; }
    .stButton>button:hover { background-color: #b3000f; color: white; }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"questions_asked": 0, "pages_used": []}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    
    lang_choice = st.radio("🌐 Choose Language / اختر اللغة", ["English", "العربية"])
    lang = LOCALES[lang_choice]
    
    st.header(lang["config"])
    
    provider_choice = st.selectbox(lang["provider"], ["Google Gemini", "OpenAI"])
    
    if provider_choice == "Google Gemini":
        model_options = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        env_key_name = "GOOGLE_API_KEY"
    else:
        model_options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        env_key_name = "OPENAI_API_KEY"
        
    model_choice = st.selectbox(lang["model"], model_options)
    api_key = st.text_input(f"{lang['api_lbl']} ({provider_choice})", type="password")
    
    st.header(lang["upload_hd"])
    
    if not api_key:
        st.info(lang["unlock_upload"])
    else:
        uploaded_files = st.file_uploader(lang["upload_lbl"], type="pdf", accept_multiple_files=True)
        
        if st.button(lang["process_btn"]):
            if not uploaded_files:
                st.warning(lang["file_warn"])
            else:
                os.environ[env_key_name] = api_key
                os.environ["GOOGLE_API_KEY"] = api_key if provider_choice == "Google Gemini" else os.environ.get("GOOGLE_API_KEY", api_key)
                
                with st.spinner(lang["thinking"]):
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(docs)
                        
                        for chunk in chunks:
                            chunk.metadata["source"] = uploaded_file.name
                        all_chunks.extend(chunks)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                    st.session_state.vectorstore = Chroma.from_documents(all_chunks, embeddings)
                    st.success(lang["success_msg"].format(len(uploaded_files), len(all_chunks)))

    st.markdown("---")
    st.header(lang["stats_hd"])
    st.metric(lang["total_q"], st.session_state.stats["questions_asked"])
    
    if st.session_state.stats["pages_used"]:
        st.subheader(lang["top_p"])
        page_counts = Counter(st.session_state.stats["pages_used"])
        for page, count in page_counts.most_common(3):
            st.write(f"- {page}: {count}")

    st.markdown("---")
    if st.session_state.messages:
        chat_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(label=lang["download_btn"], data=chat_json, file_name="gis_chat_history.json", mime="application/json")

st.title(lang["title"])
st.markdown("---")

if st.session_state.vectorstore is not None:
    st.subheader(lang["suggested_hd"])
    
    if lang_choice == "English":
        suggestions = [
            "What is the difference between raster and vector data?",
            "Explain WGS84 and EPSG:4326 coordinate systems.",
            "What is PostGIS and what is it used for?"
        ]
    else:
        suggestions = [
            "ما الفرق بين بيانات الـ Raster والـ Vector؟",
            "اشرح نظام الإحداثيات WGS84 و EPSG:4326.",
            "ما هو PostGIS وفيما يستخدم؟"
        ]
        
    col1, col2, col3 = st.columns(3)
    clicked_question = None
    if col1.button(suggestions[0]): clicked_question = suggestions[0]
    if col2.button(suggestions[1]): clicked_question = suggestions[1]
    if col3.button(suggestions[2]): clicked_question = suggestions[2]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    chat_input_query = st.chat_input(lang["ask_placeholder"])
    active_query = chat_input_query if chat_input_query else clicked_question
    
    if active_query:
        st.session_state.stats["questions_asked"] += 1
        st.session_state.messages.append({"role": "user", "content": active_query})
        
        if clicked_question:
            with st.chat_message("user"):
                st.write(active_query)
                
        with st.chat_message("assistant"):
            with st.spinner(lang["thinking"]):
                os.environ[env_key_name] = api_key
                
                docs = st.session_state.vectorstore.similarity_search(active_query, k=4)
                context = "\n\n".join([d.page_content for d in docs])
                
                for d in docs:
                    page_num = d.metadata.get("page", "Unknown")
                    file_name = d.metadata.get("source", "Unknown")
                    st.session_state.stats["pages_used"].append(f"{file_name} (Pg {page_num})")
                
                prompt = f"""You are a helpful GIS assistant. Use the following context to answer the user's question. 
If the user asks in Arabic, answer in Arabic. If English, answer in English.
If you cannot find the answer in the context, clearly state that you do not have enough information.

Context:
{context}

Question: {active_query}

Answer:"""
                
                if provider_choice == "Google Gemini":
                    model_instance = ChatGoogleGenerativeAI(model=model_choice, temperature=0.3)
                else:
                    model_instance = ChatOpenAI(model=model_choice, temperature=0.3)
                    
                response = model_instance.invoke(prompt)
                answer = response.content
                
                st.write(answer)
                
                with st.expander(lang["sources_hd"]):
                    for i, doc in enumerate(docs, 1):
                        source_file = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Source {i}** - `{source_file}` (Page {page}):")
                        st.info(doc.page_content)
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if clicked_question:
            st.rerun()
else:
    if not api_key:
        st.info(lang["step_1_msg"])
    else:
        st.info(lang["step_2_msg"])