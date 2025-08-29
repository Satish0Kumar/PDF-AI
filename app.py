# app.py - Enhanced PDF QA System with Advanced Features
import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure the application
st.set_page_config(page_title="Advanced PDF QA System", layout="wide")
st.title("📚 Advanced PDF Question Answering System")
st.write("Upload PDF files and ask questions with intelligent response handling.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "document_summaries" not in st.session_state:
    st.session_state.document_summaries = {}
if "use_general_knowledge" not in st.session_state:
    st.session_state.use_general_knowledge = False
if "current_model" not in st.session_state:
    st.session_state.current_model = "models/gemini-1.5-flash"

# Load environment variables for local development
load_dotenv()

# Configure Gemini - Proper initialization
api_key = None

# First try to get the key from Streamlit Secrets (for deployment)
try:
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.sidebar.success("✅ API key loaded from Streamlit secrets")
except Exception as e:
    pass  # Silent fail, we'll try other methods

# If not found in secrets, try .env file (for local development)
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")  # <- FIXED THE TYPO HERE!
    if api_key:
        st.sidebar.info("ℹ️ API key loaded from .env file")

# If still not found, show error
if not api_key:
    st.error("""
    ❌ GEMINI_API_KEY not found. 
    
    **For local development:** Create a `.env` file with `GEMINI_API_KEY=your_key_here`
    **For deployment:** Add your key to Streamlit Cloud secrets
    """)
else:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")


# Available Gemini models (in order of preference)
GEMINI_MODELS = [
    "models/gemini-1.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash"
]

def switch_model():
    """Switch to the next available model if current one fails"""
    current_index = GEMINI_MODELS.index(st.session_state.current_model)
    next_index = (current_index + 1) % len(GEMINI_MODELS)
    st.session_state.current_model = GEMINI_MODELS[next_index]
    st.sidebar.info(f"Switched to model: {st.session_state.current_model}")
    return st.session_state.current_model

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF {pdf_file.name}: {str(e)}")
        return ""

def generate_document_summary(text, filename):
    """Generate a summary for each document"""
    try:
        # Take first 2000 characters for summary
        preview = text[:2000] + "..." if len(text) > 2000 else text
        
        prompt = f"""
        Please provide a concise 2-3 sentence summary of this document:
        
        {preview}
        
        Document: {filename}
        """
        
        model = genai.GenerativeModel(st.session_state.current_model)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

def process_pdfs(pdf_files):
    """Process all uploaded PDF files"""
    all_text = ""
    for pdf_file in pdf_files:
        with st.spinner(f"Processing {pdf_file.name}..."):
            text = extract_text_from_pdf(pdf_file)
            if text:
                all_text += text + "\n\n"
                # Generate and store summary
                summary = generate_document_summary(text, pdf_file.name)
                st.session_state.document_summaries[pdf_file.name] = summary
    return all_text

def chunk_text(text):
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks"""
    try:
        embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def analyze_question_style(question):
    """Analyze the question to determine the desired response style"""
    question_lower = question.lower()
    
    # Check for detail requests
    detail_keywords = ["in detail", "detailed", "explain", "how does", "how to", "step by step"]
    if any(keyword in question_lower for keyword in detail_keywords):
        return "detailed"
    
    # Check for summary requests
    summary_keywords = ["summary", "summarize", "brief", "concise", "in brief", "in short"]
    if any(keyword in question_lower for keyword in summary_keywords):
        return "concise"
    
    # Default to medium
    return "medium"

def get_gemini_response(question, context, use_general_knowledge=False, style="medium"):
    """Get response from Gemini AI with enhanced capabilities"""
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Determine response style instructions
            style_instructions = {
                "detailed": "Provide a comprehensive, detailed explanation with examples if possible.",
                "concise": "Provide a concise, to-the-point answer without unnecessary details.",
                "medium": "Provide a balanced answer with adequate explanation but not overly verbose."
            }
            
            if use_general_knowledge:
                prompt = f"""
                {style_instructions[style]}
                
                Please answer the following question based on your general knowledge:
                
                Question: {question}
                
                Answer:
                """
            else:
                prompt = f"""
                {style_instructions[style]}
                
                Based on the following context from uploaded documents, please answer the question.
                If the concept is mentioned in the context but not fully explained, you can enhance the answer with your knowledge.
                
                Context:
                {context}
                
                Question: {question}
                
                Please provide a helpful and accurate answer:
                """
            
            model = genai.GenerativeModel(st.session_state.current_model)
            response = model.generate_content(prompt)
            return response.text, False  # False means no error
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "limit" in error_msg.lower() or "429" in error_msg:
                retry_count += 1
                if retry_count <= max_retries:
                    st.sidebar.warning(f"Rate limit hit. Switching model (attempt {retry_count}/{max_retries})...")
                    switch_model()
                else:
                    return f"Error: API quota exceeded. Please try again later.", True
            else:
                return f"Error getting response: {error_msg}", True
    
    return "Unexpected error occurred.", True

def find_relevant_context(question, vector_store, k=3):
    """Find relevant context from vector store"""
    if vector_store is None:
        return "No documents processed yet."
    
    try:
        # Search for similar documents
        docs = vector_store.similarity_search(question, k=k)
        # Combine the content
        context = "\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        return f"Error searching documents: {str(e)}"

def is_query_in_context(question, context):
    """Check if the question is related to the context"""
    if not context or "No documents processed yet" in context or "Error searching" in context:
        return False
    
    # Simple check: see if any meaningful words from the question appear in the context
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "in", "on", "at", "to", "for", "of", "and", "or", "what", "how", "why"}
    question_words = question_words - stop_words
    context_words = context_words - stop_words
    
    # If we have at least 2 matching meaningful words, consider it relevant
    return len(question_words & context_words) >= 2

# Sidebar for PDF upload and document management
with st.sidebar:
    st.header("📁 Document Management")
    
    pdf_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Select one or more PDF files to process"
    )
    
    if st.button("🔄 Process Documents") and pdf_files:
        with st.spinner("Processing documents..."):
            # Extract text from PDFs
            text = process_pdfs(pdf_files)
            
            if text.strip():
                # Split text into chunks
                chunks = chunk_text(text)
                
                # Create vector store
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.processed = True
                
                st.success(f"✅ Processed {len(pdf_files)} PDF(s)!")
                st.info(f"Found {len(chunks)} text chunks for searching.")
            else:
                st.error("No text could be extracted from the PDFs.")
    
    # Display document summaries
    if st.session_state.document_summaries:
        st.divider()
        st.header("📄 Document Summaries")
        for filename, summary in st.session_state.document_summaries.items():
            with st.expander(f"{filename}"):
                st.write(summary)
    
    # Model information
    st.divider()
    st.header("⚙️ Model Settings")
    st.info(f"Current model: {st.session_state.current_model}")
    if st.button("🔄 Switch Model"):
        switch_model()

# Main chat area
st.header("💬 Chat with Your Documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show if general knowledge was used
        if message.get("general_knowledge", False):
            st.caption("🔆 Answered using general knowledge")

# Question input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.write(question)
    
    if not st.session_state.processed:
        with st.chat_message("assistant"):
            st.write("Please upload and process PDF documents first using the sidebar.")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": "Please upload and process PDF documents first using the sidebar."
        })
    else:
        # Find relevant context
        with st.spinner("🔍 Searching documents..."):
            context = find_relevant_context(question, st.session_state.vector_store)
        
        # Check if query is related to context
        query_in_context = is_query_in_context(question, context)
        
        # Analyze question style
        response_style = analyze_question_style(question)
        
        # Generate response based on context
        with st.spinner("🤔 Generating answer..."):
            answer, error = get_gemini_response(question, context, False, response_style)
            
            # If we got an error or the answer indicates no context found
            if error or (not query_in_context and "couldn't find this information" in answer.lower()):
                # Store the question for potential general knowledge answer
                st.session_state.last_question = question
                st.session_state.last_style = response_style
                st.session_state.show_general_knowledge_button = True
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "general_knowledge": False
                })
                
                with st.chat_message("assistant"):
                    st.write(answer)
            else:
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "general_knowledge": False
                })
                
                with st.chat_message("assistant"):
                    st.write(answer)
                
                st.session_state.show_general_knowledge_button = False

# Show general knowledge button if needed
if hasattr(st.session_state, 'show_general_knowledge_button') and st.session_state.show_general_knowledge_button:
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.warning("This information wasn't found in your documents. Would you like to ask the AI using its general knowledge?")
    
    with col2:
        if st.button("🤖 Use AI General Knowledge"):
            with st.spinner("Consulting general knowledge..."):
                general_answer, error = get_gemini_response(
                    st.session_state.last_question, 
                    "", 
                    True, 
                    st.session_state.last_style
                )
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": general_answer,
                    "general_knowledge": True
                })
                
                st.session_state.show_general_knowledge_button = False
                st.rerun()

# Export conversation
if st.session_state.chat_history:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Conversation"):
            export_data = {
                "export_date": datetime.now().isoformat(),
                "chat_history": st.session_state.chat_history,
                "document_summaries": st.session_state.document_summaries
            }
            
            json_data = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="💾 Download as JSON",
                data=json_data,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            if hasattr(st.session_state, 'show_general_knowledge_button'):
                st.session_state.show_general_knowledge_button = False
            st.rerun()

# Instructions
with st.expander("ℹ️ How to use this advanced system"):
    st.markdown("""
    ## 📖 Enhanced Features Guide
    
    **Intelligent Response Handling:**
    - The system first tries to answer from your documents
    - If the concept exists in documents but needs more detail, AI enhances the answer
    - If the concept isn't in documents, you can choose to use AI's general knowledge
    
    **Adaptive Response Style:**
    - Use words like "in detail" or "explain" for comprehensive answers
    - Use words like "summary" or "brief" for concise answers
    - Otherwise, you'll get balanced responses
    
    **Model Management:**
    - Automatically switches between Gemini models if limits are hit
    - Manually switch models using the sidebar button
    
    ## 💡 Usage Tips
    1. Upload and process your PDF documents
    2. Ask questions naturally - the system detects your preferred detail level
    3. If information isn't found in documents, use the general knowledge option
    4. Export conversations for future reference
    """)

# Footer
st.divider()
st.caption("Powered by Google Gemini AI • Built with Streamlit • Advanced PDF QA System")

if __name__ == "__main__":
    pass