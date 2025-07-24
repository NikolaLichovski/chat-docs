import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"


def check_ollama_running():
    """Check if Ollama is running and has models available"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return len(models) > 0, models
        return False, []
    except:
        return False, []


def query_rag(query_text: str):
    """Query the RAG system with a proper retrieval chain"""

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        return "Database not found. Please run create_database.py first to create the vector database.", []

    # Initialize embeddings (same as used for creation)
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load vector database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Test retrieval first
    docs = db.similarity_search(query_text, k=5)
    if not docs:
        return "No relevant documents found for your query.", []

    print(f"Found {len(docs)} relevant documents")

    # Try Ollama first (best option if available)
    ollama_running, models = check_ollama_running()

    if ollama_running and models:
        try:
            # Use Ollama with a good model
            model_name = "llama3.2" if any("llama3.2" in m['name'] for m in models) else models[0]['name']
            print(f"Using Ollama model: {model_name}")

            llm = Ollama(
                model=model_name,
                temperature=0.3,
                top_p=0.9,
            )

            # Create a proper RAG chain
            template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always provide a complete, well-structured answer based on the context.

Context:
{context}

Question: {question}

Answer: """

            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=db.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )

            result = qa_chain.invoke({"query": query_text})
            response_text = result["result"]
            sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

            return response_text, list(set(sources))  # Remove duplicates

        except Exception as e:
            print(f"Ollama failed: {e}")

    # Fallback to HuggingFace Transformers (local)
    try:
        print("Using local Hugging Face model...")
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        # Use a better local model for text generation
        model_name = "microsoft/DialoGPT-medium"

        # Try to use a more capable model if possible
        try:
            model_name = "microsoft/DialoGPT-large"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except:
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs[:3]])

        # Create a comprehensive prompt
        prompt = f"""Based on the following context, please provide a detailed and complete answer to the question.

Context:
{context}

Question: {query_text}

Please provide a thorough answer based on the context above:
"""

        # Use text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=len(prompt.split()) + 150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        result = generator(prompt)
        full_response = result[0]['generated_text']

        # Extract just the answer part
        if "Please provide a thorough answer based on the context above:" in full_response:
            response_text = full_response.split("Please provide a thorough answer based on the context above:")[
                -1].strip()
        else:
            response_text = full_response[len(prompt):].strip()

        if not response_text or len(response_text) < 10:
            raise Exception("Generated response too short")

        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        return response_text, list(set(sources))

    except Exception as e:
        print(f"Local Hugging Face model failed: {e}")

    # Final fallback - structured context response
    print("Using structured context fallback...")

    # Group relevant chunks and create a structured response
    context_chunks = []
    sources = []

    for doc in docs[:3]:
        chunk_text = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown")

        # Clean up the chunk
        if len(chunk_text) > 50:
            context_chunks.append(chunk_text)
            sources.append(source)

    if context_chunks:
        # Create a structured response from the most relevant chunks
        response_parts = []
        query_lower = query_text.lower()

        for chunk in context_chunks:
            # Find sentences that seem most relevant to the query
            sentences = [s.strip() for s in chunk.split('.') if s.strip()]
            relevant_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Score relevance based on query word overlap
                query_words = [w for w in query_lower.split() if len(w) > 3]
                matches = sum(1 for word in query_words if word in sentence_lower)

                if matches > 0:
                    relevant_sentences.append((sentence, matches))

            # Sort by relevance and take best sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)

            if relevant_sentences:
                best_sentences = [s[0] for s in relevant_sentences[:2]]
                response_parts.extend(best_sentences)

        # Combine into a coherent response
        if response_parts:
            # Remove duplicates while preserving order
            seen = set()
            unique_parts = []
            for part in response_parts:
                if part not in seen:
                    seen.add(part)
                    unique_parts.append(part)

            response_text = '. '.join(unique_parts[:3])
            if not response_text.endswith('.'):
                response_text += '.'
        else:
            response_text = f"Based on the documents, here's the relevant information: {context_chunks[0][:400]}..."
    else:
        response_text = "I found some relevant information but couldn't extract a clear answer from the documents."

    return response_text, list(set(sources))


def main():
    """CLI interface for testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    response, sources = query_rag(args.query_text)

    print(f"Response: {response}")
    print(f"Sources: {sources}")


if __name__ == "__main__":
    main()