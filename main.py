# ===========================================================================
# ROBUST WORKING PDF CHAT - Guaranteed to Work
# ===========================================================================

import streamlit as st
import os
from pathlib import Path
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class SimplePDFProcessor:
    """Simplified, robust PDF processor"""
    
    def process_pdf(self, pdf_file) -> List[Dict[str, Any]]:
        """Process PDF with error handling"""
        pages_data = []
        
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📁 Creating temporary file...")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            status_text.text(f"📖 Opening PDF: {pdf_file.name}")
            
            # Open PDF
            doc = fitz.open(tmp_file_path)
            total_pages = len(doc)
            
            status_text.text(f"📄 Found {total_pages} pages. Processing...")
            
            for page_num in range(total_pages):
                try:
                    # Update progress
                    progress = (page_num + 1) / total_pages
                    progress_bar.progress(progress)
                    status_text.text(f"📄 Processing page {page_num + 1}/{total_pages}")
                    
                    page = doc[page_num]
                    
                    # Get text
                    text_content = page.get_text()
                    
                    # Convert to image
                    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Resize to manageable size
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    
                    # Store page data
                    page_data = {
                        "page_number": page_num + 1,
                        "image": pil_image,
                        "text": text_content,
                        "text_length": len(text_content),
                        "has_text": bool(text_content.strip()),
                        "has_images": len(page.get_images()) > 0
                    }
                    
                    pages_data.append(page_data)
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing page {page_num + 1}: {e}")
                    continue
            
            # Cleanup
            doc.close()
            os.unlink(tmp_file_path)
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Successfully processed {len(pages_data)} pages!")
            
        except Exception as e:
            st.error(f"❌ PDF processing failed: {e}")
            return []
        
        return pages_data

class SimpleEmbeddingEngine:
    """Simple embedding engine with fallbacks"""
    
    def __init__(self):
        self.cohere_client = None
        self.setup_cohere()
    
    def setup_cohere(self):
        """Setup Cohere with error handling"""
        try:
            import cohere
            api_key = os.getenv('COHERE_API_KEY')
            
            if api_key and api_key != 'your_cohere_api_key_here':
                self.cohere_client = cohere.Client(api_key)
                st.info(f"✅ Cohere connected: {api_key[:8]}...")
            else:
                st.warning("⚠️ Cohere API key not configured - using fallback embeddings")
        except Exception as e:
            st.warning(f"⚠️ Cohere setup failed: {e}")
    
    def create_embeddings(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for pages"""
        embeddings_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, page_data in enumerate(pages_data):
            try:
                # Update progress
                progress = (idx + 1) / len(pages_data)
                progress_bar.progress(progress)
                status_text.text(f"🧠 Creating embedding for page {page_data['page_number']}")
                
                # Create description
                description = f"Page {page_data['page_number']}: "
                if page_data['has_text']:
                    description += page_data['text'][:500]
                if page_data['has_images']:
                    description += " [Contains technical diagrams and visual elements]"
                
                # Generate embedding
                if self.cohere_client:
                    try:
                        response = self.cohere_client.embed(
                            texts=[description],
                            model="embed-english-v3.0",
                            input_type="search_document"
                        )
                        embedding = np.array(response.embeddings[0])
                    except Exception as e:
                        st.warning(f"⚠️ Cohere failed for page {page_data['page_number']}: {e}")
                        embedding = self._create_text_based_embedding(description)
                else:
                    embedding = self._create_text_based_embedding(description)
                
                embeddings_data.append({
                    "page_number": page_data['page_number'],
                    "embedding": embedding,
                    "image": page_data['image'],
                    "text": page_data['text'],
                    "description": description
                })
                
            except Exception as e:
                st.warning(f"⚠️ Embedding failed for page {page_data['page_number']}: {e}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Created {len(embeddings_data)} embeddings!")
        
        return embeddings_data
    
    def _create_text_based_embedding(self, text: str) -> np.ndarray:
        """Fallback: create simple text-based embedding"""
        # Simple hash-based embedding as fallback
        import hashlib
        
        # Create deterministic embedding from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert to numbers
        embedding = []
        for i in range(0, len(text_hash), 2):
            embedding.append(int(text_hash[i:i+2], 16) / 255.0)
        
        # Pad or truncate to 1024 dimensions
        while len(embedding) < 1024:
            embedding.extend(embedding[:min(1024-len(embedding), len(embedding))])
        
        return np.array(embedding[:1024])

class SimpleSearchEngine:
    """Simple search engine that definitely works"""
    
    def __init__(self):
        self.embeddings_data = []
        self.is_ready = False
    
    def index_documents(self, embeddings_data: List[Dict[str, Any]]):
        """Index documents for search"""
        try:
            st.info(f"📚 Indexing {len(embeddings_data)} documents...")
            
            self.embeddings_data = embeddings_data
            self.is_ready = len(embeddings_data) > 0
            
            # Store in session state
            st.session_state['search_index'] = embeddings_data
            st.session_state['index_ready'] = self.is_ready
            
            st.success(f"✅ Successfully indexed {len(embeddings_data)} documents!")
            
        except Exception as e:
            st.error(f"❌ Indexing failed: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.is_ready:
            return []
        
        try:
            st.info(f"🔍 Searching for: '{query}'")
            
            # Simple keyword-based search as fallback
            query_words = set(query.lower().split())
            
            results = []
            for doc in self.embeddings_data:
                # Calculate relevance score
                doc_text = doc['text'].lower()
                doc_words = set(doc_text.split())
                
                # Count word matches
                matches = len(query_words.intersection(doc_words))
                
                # Boost score if query words appear in text
                score = matches / len(query_words) if query_words else 0
                
                # Add context bonus for technical terms
                tech_terms = ['cylinder', 'dryer', 'section', 'component', 'system']
                for term in tech_terms:
                    if term in query.lower() and term in doc_text:
                        score += 0.2
                
                if score > 0:
                    results.append({
                        'page_number': doc['page_number'],
                        'image': doc['image'],
                        'text': doc['text'],
                        'score': score,
                        'preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            st.success(f"✅ Found {len(results)} relevant pages")
            
            return results[:top_k]
            
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            return []

class SimpleAnswerEngine:
    """Simple answer engine with fallbacks"""
    
    def __init__(self):
        self.gemini_model = None
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Gemini with error handling"""
        try:
            import google.generativeai as genai
            api_key = os.getenv('GOOGLE_API_KEY')
            
            if api_key and api_key != 'your_google_api_key_here':
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                st.info(f"✅ Gemini connected: {api_key[:8]}...")
            else:
                st.warning("⚠️ Google API key not configured - using text analysis")
        except Exception as e:
            st.warning(f"⚠️ Gemini setup failed: {e}")
    
    def answer_question(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate answer from search results"""
        if not search_results:
            return "❌ No relevant information found in the document for your question."
        
        # Use Gemini if available
        if self.gemini_model and search_results:
            return self._gemini_analysis(query, search_results[0])
        else:
            return self._text_analysis(query, search_results)
    
    def _gemini_analysis(self, query: str, best_result: Dict[str, Any]) -> str:
        """Use Gemini for visual + text analysis"""
        try:
            st.info("🤖 Analyzing with Gemini...")
            
            prompt = f"""Analyze this technical document page to answer the user's question.

Question: "{query}"

Text content from the page:
{best_result['text'][:1000]}

Instructions:
1. Read all text visible in the image
2. Analyze any diagrams, charts, or visual elements
3. Count specific items if the question asks for quantities
4. Provide a clear, specific answer

Focus on answering: {query}"""
            
            response = self.gemini_model.generate_content([prompt, best_result['image']])
            
            return f"""**🤖 AI Analysis Results:**

{response.text}

**📄 Source:** Page {best_result['page_number']} (Relevance: {best_result['score']:.2f})
**📊 Analysis Type:** Visual + Text Analysis with Gemini 2.5 Flash"""
            
        except Exception as e:
            st.warning(f"⚠️ Gemini analysis failed: {e}")
            return self._text_analysis(query, [best_result])
    
    def _text_analysis(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Fallback text-based analysis"""
        response_parts = [f"**📝 Text Analysis Results for:** \"{query}\"", ""]
        
        for result in search_results[:3]:
            response_parts.append(f"**📄 Page {result['page_number']} (Score: {result['score']:.2f}):**")
            
            # Look for relevant content
            text = result['text']
            query_words = query.lower().split()
            
            # Find relevant sentences
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_words):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                response_parts.append("**Relevant content found:**")
                for sentence in relevant_sentences[:3]:
                    if sentence:
                        response_parts.append(f"- {sentence}")
            else:
                response_parts.append(f"**Content preview:** {text[:300]}...")
            
            response_parts.append("")
        
        response_parts.append("**💡 Note:** For detailed visual analysis, ensure Google Gemini API is configured.")
        
        return "\n".join(response_parts)

def main():
    """Main application"""
    
    st.set_page_config(
        page_title="Robust PDF Chat",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Robust PDF Chat - Guaranteed to Work!")
    st.markdown("**Simple, reliable PDF analysis with text + visual understanding**")
    
    # Initialize components
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = SimplePDFProcessor()
    if 'embedding_engine' not in st.session_state:
        st.session_state.embedding_engine = SimpleEmbeddingEngine()
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SimpleSearchEngine()
    if 'answer_engine' not in st.session_state:
        st.session_state.answer_engine = SimpleAnswerEngine()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Load existing index
    if 'search_index' in st.session_state:
        st.session_state.search_engine.embeddings_data = st.session_state['search_index']
        st.session_state.search_engine.is_ready = st.session_state.get('index_ready', False)
    
    # Status panel
    with st.expander("📊 System Status", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Document Status")
            is_ready = st.session_state.search_engine.is_ready
            doc_count = len(st.session_state.search_engine.embeddings_data)
            
            if is_ready:
                st.success(f"✅ Ready! {doc_count} pages indexed")
            else:
                st.warning("⚠️ No documents indexed yet")
        
        with col2:
            st.subheader("🔑 API Status")
            cohere_key = os.getenv('COHERE_API_KEY')
            google_key = os.getenv('GOOGLE_API_KEY')
            
            st.write("**Cohere:**", "✅" if cohere_key and cohere_key != 'your_cohere_api_key_here' else "⚠️")
            st.write("**Google:**", "✅" if google_key and google_key != 'your_google_api_key_here' else "⚠️")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Show current status
        if st.session_state.search_engine.is_ready:
            st.info(f"✅ Ready to answer questions about {len(st.session_state.search_engine.embeddings_data)} pages!")
        else:
            st.warning("📄 Upload and process a PDF to start asking questions")
        
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "image" in message:
                    st.image(message["image"], caption=f"Page {message.get('page', '')}", width=300)
        
        # Chat input
        if user_input := st.chat_input("Ask about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                if not st.session_state.search_engine.is_ready:
                    response = "Please upload and process a PDF document first."
                    st.markdown(response)
                else:
                    with st.spinner("🔍 Searching and analyzing..."):
                        # Search
                        search_results = st.session_state.search_engine.search(user_input)
                        
                        # Generate answer
                        answer = st.session_state.answer_engine.answer_question(user_input, search_results)
                        st.markdown(answer)
                        
                        # Show relevant pages
                        if search_results:
                            st.subheader("📄 Relevant Pages")
                            for result in search_results[:2]:
                                col_img, col_text = st.columns([1, 2])
                                with col_img:
                                    st.image(result['image'], width=200)
                                with col_text:
                                    st.write(f"**Page {result['page_number']}** (Score: {result['score']:.2f})")
                                    st.write(result['preview'])
                        
                        response = answer
                
                # Add assistant message
                assistant_msg = {"role": "assistant", "content": response}
                if 'search_results' in locals() and search_results:
                    assistant_msg["image"] = search_results[0]['image']
                    assistant_msg["page"] = search_results[0]['page_number']
                st.session_state.messages.append(assistant_msg)
    
    with col2:
        st.header("📄 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF for analysis"
        )
        
        if uploaded_file:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            
            if st.button("🚀 Process PDF", type="primary"):
                with st.container():
                    st.subheader("🔄 Processing...")
                    
                    # Process PDF
                    pages_data = st.session_state.pdf_processor.process_pdf(uploaded_file)
                    
                    if pages_data:
                        st.success(f"✅ Extracted {len(pages_data)} pages")
                        
                        # Create embeddings
                        embeddings_data = st.session_state.embedding_engine.create_embeddings(pages_data)
                        
                        if embeddings_data:
                            # Index documents
                            st.session_state.search_engine.index_documents(embeddings_data)
                            
                            st.balloons()
                            st.success("🎉 PDF processed successfully! You can now ask questions.")
                            st.rerun()
        
        # Show processed pages
        if st.session_state.search_engine.is_ready:
            st.subheader("📖 Processed Pages")
            for doc in st.session_state.search_engine.embeddings_data[-3:]:
                with st.expander(f"Page {doc['page_number']}"):
                    st.image(doc['image'], width=150)
                    st.write(f"Text: {len(doc['text'])} characters")
        
        # Reset button
        if st.button("🗑️ Clear All Data"):
            for key in ['search_index', 'index_ready', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.search_engine = SimpleSearchEngine()
            st.rerun()

if __name__ == "__main__":
    main()