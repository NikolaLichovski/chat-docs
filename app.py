import streamlit as st
import os
import shutil
from create_database import generate_data_store
from query_data import query_rag
from document_loader import DocumentLoader

# Page config
st.set_page_config(
    page_title="Easy Chat for Your Docs",
    page_icon="📚",
    layout="wide"
)

# Constants
DATA_PATH = "data/docs"
CHROMA_PATH = "chroma"


def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    st.title("📚 Easy Chat for Your Docs")
    st.markdown("Upload your documents and chat with them using AI!")

    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['txt', 'md', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: TXT, MD, PDF, DOCX"
        )

        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")

            if st.button("💾 Save & Process Documents"):
                with st.spinner("Saving and processing documents..."):
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        save_uploaded_file(uploaded_file)

                    # Generate database
                    try:
                        generate_data_store()
                        st.success("✅ Documents processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

        # Show existing files
        st.subheader("📄 Current Documents")
        if os.path.exists(DATA_PATH):
            files = os.listdir(DATA_PATH)
            if files:
                for file in files:
                    st.text(f"• {file}")
            else:
                st.info("No documents uploaded yet")
        else:
            st.info("No documents uploaded yet")

        # Ollama status
        st.subheader("🤖 AI Model Status")

        # Check Ollama status
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    st.success(f"✅ Ollama running with {len(models)} model(s)")
                    with st.expander("Available models"):
                        for model in models:
                            st.text(f"• {model['name']}")
                else:
                    st.warning("⚠️ Ollama running but no models installed")
                    st.info("Run: `ollama pull llama3.2` to install a model")
            else:
                st.error("❌ Ollama not responding")
        except:
            st.warning("⚠️ Ollama not running - will use fallback models")
            st.info("Install Ollama for better responses!")

        # Database status
        st.subheader("🗄️ Database Status")
        if os.path.exists(CHROMA_PATH):
            st.success("✅ Vector database ready")
        else:
            st.warning("⚠️ No database found. Upload and process documents first.")

        # Clear database button
        if st.button("🗑️ Clear All Data"):
            if os.path.exists(DATA_PATH):
                shutil.rmtree(DATA_PATH)
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            st.success("All data cleared!")
            st.rerun()

    # Main chat interface
    st.header("💬 Chat with Your Documents")

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        st.warning("⚠️ Please upload and process documents first using the sidebar.")
        return

    # Query input
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic discussed in the documents?",
        help="Type your question and press Enter"
    )

    if query:
        with st.spinner("Searching and generating response..."):
            try:
                response, sources = query_rag(query)

                # Display response
                st.subheader("📝 Response")
                st.write(response)

                # Display sources
                st.subheader("📚 Sources")
                if sources:
                    for i, source in enumerate(sources, 1):
                        # Clean up source path for display
                        display_source = os.path.basename(source) if source != "Unknown" else "Unknown"
                        st.text(f"{i}. {display_source}")
                else:
                    st.text("No sources found")

            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ## Setup Steps:
        1. **Install Ollama** (recommended for best results):
           - Go to [ollama.ai](https://ollama.ai) and download
           - Run: `ollama pull llama3.2`

        2. **Upload Documents**: Use the sidebar to upload your documents (TXT, MD, PDF, DOCX)

        3. **Process**: Click "Save & Process Documents" to create the vector database

        4. **Chat**: Ask questions about your documents in the text input above

        ## Model Priority:
        1. **Ollama** (best) - Smart local models like Llama 3.2
        2. **Hugging Face local** (fallback) - Smaller models
        3. **Context extraction** (last resort) - Direct text snippets

        **Supported File Types:**
        - `.txt` - Plain text files
        - `.md` - Markdown files  
        - `.pdf` - PDF documents
        - `.docx` - Word documents

        **Why Ollama?**
        - Much smarter responses
        - Completely free and private
        - No API keys needed
        - Works offline
        """)

    # Show current model being used
    if query:
        with st.sidebar:
            st.info("💡 For better responses, install Ollama and run `ollama pull llama3.2`")


if __name__ == "__main__":
    main()