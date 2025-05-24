# UI HELPERS - utils/ui_helpers.py
# ===========================================================================

import streamlit as st
from typing import List, Dict, Any, Optional
from PIL import Image
import io

class UIComponents:
    """
    Streamlit UI components for the multimodal PDF chat application
    Provides reusable interface elements with consistent styling
    """
    
    @staticmethod
    def render_upload_section():
        """
        Render file upload section in sidebar
        """
        st.header("📁 Upload Documents")
        
        # File uploader with multiple format support
        uploaded_files = st.file_uploader(
            "Choose PDF or image files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload PDFs or images for visual analysis"
        )
        
        if uploaded_files:
            UIComponents._process_uploaded_files(uploaded_files)
    
    @staticmethod
    def _process_uploaded_files(uploaded_files):
        """
        Process and display uploaded files
        """
        st.subheader("📋 Uploaded Files")
        
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Size:** {file.size / 1024:.1f} KB")
                    st.write(f"**Type:** {file.type}")
                
                with col2:
                    if st.button(f"Process", key=f"process_{file.name}"):
                        UIComponents._handle_file_processing(file)
    
    @staticmethod
    def _handle_file_processing(file):
        """
        Handle individual file processing
        """
        with st.spinner(f"Processing {file.name}..."):
            if file.type == "application/pdf":
                # Process PDF
                pdf_processor = st.session_state.pdf_processor
                image_data = pdf_processor.convert_pdf_to_images(file)
                
                if image_data:
                    # Generate embeddings
                    embedding_engine = st.session_state.embedding_engine
                    embeddings_data = embedding_engine.generate_image_embeddings(image_data)
                    
                    # Add to retrieval index
                    retrieval_engine = st.session_state.retrieval_engine
                    retrieval_engine.add_documents(embeddings_data)
                    
                    # Update session state
                    if 'processed_documents' not in st.session_state:
                        st.session_state.processed_documents = []
                    
                    st.session_state.processed_documents.append({
                        "filename": file.name,
                        "type": "pdf",
                        "pages": len(image_data),
                        "processed_at": "now"
                    })
                    
                    st.success(f"✅ Processed {len(image_data)} pages from {file.name}")
                else:
                    st.error("Failed to process PDF")
            
            else:
                # Process individual image
                pdf_processor = st.session_state.pdf_processor
                image_data = pdf_processor.process_uploaded_image(file)
                
                if image_data:
                    # Process as single image
                    embedding_engine = st.session_state.embedding_engine
                    embeddings_data = embedding_engine.generate_image_embeddings([image_data])
                    
                    retrieval_engine = st.session_state.retrieval_engine
                    retrieval_engine.add_documents(embeddings_data)
                    
                    st.success(f"✅ Processed image {file.name}")
    
    @staticmethod
    def render_sample_documents():
        """
        Render sample documents section
        """
        st.header("📊 Sample Documents")
        
        sample_docs = [
            "Financial Report Q4 2023",
            "Technical Architecture Diagram", 
            "Market Analysis Charts",
            "Product Roadmap Visualization"
        ]
        
        for doc in sample_docs:
            if st.button(f"📄 Load {doc}", key=f"sample_{doc}"):
                st.info(f"Loading {doc}... (This would load a pre-configured sample)")
    
    @staticmethod
    def render_processing_status():
        """
        Render document processing status
        """
        st.header("📈 Processing Status")
        
        # Get retrieval engine stats
        if hasattr(st.session_state, 'retrieval_engine'):
            stats = st.session_state.retrieval_engine.get_index_stats()
            
            st.metric("Documents Indexed", stats["total_documents"])
            st.metric("Index Status", "Ready" if stats["is_trained"] else "Empty")
            
            if stats["total_documents"] > 0:
                st.success("✅ Ready for questions!")
            else:
                st.warning("⏳ Upload documents to start")
        
        # Show processed documents
        if 'processed_documents' in st.session_state and st.session_state.processed_documents:
            st.subheader("📋 Processed Files")
            for doc in st.session_state.processed_documents[-3:]:  # Show last 3
                st.write(f"• {doc['filename']} ({doc['pages']} pages)")
    
    @staticmethod
    def render_settings_panel():
        """
        Render application settings panel
        """
        st.header("⚙️ Settings")
        
        with st.expander("Search Settings"):
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum similarity score for results"
            )
            
            max_results = st.slider(
                "Max Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of pages to analyze"
            )
        
        with st.expander("Analysis Settings"):
            analysis_mode = st.selectbox(
                "Analysis Mode",
                ["General", "Financial", "Technical", "Comparative"],
                help="Specialized analysis for different document types"
            )
            
            include_confidence = st.checkbox(
                "Show Confidence Scores",
                value=True,
                help="Display similarity scores with results"
            )
        
        if st.button("🗑️ Clear All Data"):
            if st.session_state.get('retrieval_engine'):
                st.session_state.retrieval_engine.clear_index()
                st.session_state.processed_documents = []
                st.rerun()
    
    @staticmethod
    def render_chat_interface():
        """
        Render main chat interface
        """
        st.header("💬 Ask Questions About Your Documents")
        
        # Sample questions
        st.subheader("💡 Try These Questions:")
        sample_questions = [
            "What are the key financial metrics shown in the charts?",
            "Show me revenue trends over time",
            "What technical components are illustrated in the diagrams?",
            "Compare performance metrics across different periods",
            "What are the main insights from the visual data?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"💭 {question}", key=f"sample_q_{i}"):
                    UIComponents._handle_sample_question(question)
        
        st.divider()
        
        # Chat messages display
        UIComponents._display_chat_messages()
        
        # Chat input
        UIComponents._render_chat_input()
    
    @staticmethod
    def _display_chat_messages():
        """
        Display chat message history
        """
        if 'messages' in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Display analyzed images if present
                    if "analyzed_images" in message:
                        UIComponents._display_analyzed_images(message["analyzed_images"])
    
    @staticmethod
    def _display_analyzed_images(analyzed_images: List[Dict[str, Any]]):
        """
        Display analyzed images with metadata
        """
        st.subheader("🔍 Analyzed Visual Content")
        
        for img_data in analyzed_images:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(
                    img_data["image"], 
                    caption=f"Page {img_data['page_number']}"
                )
            
            with col2:
                st.write(f"**Page:** {img_data['page_number']}")
                st.write(f"**Relevance:** {img_data['similarity_score']:.3f}")
                if "visual_elements" in img_data:
                    st.write(f"**Elements:** {', '.join(img_data['visual_elements'])}")
    
    @staticmethod
    def _render_chat_input():
        """
        Render chat input field and handle user queries
        """
        if query := st.chat_input("Ask a question about the visual content in your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Process query
            UIComponents._process_user_query(query)
    
    @staticmethod
    def _process_user_query(query: str):
        """
        Process user query and generate response
        """
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing visual content..."):
                response_data = UIComponents._generate_response(query)
                
                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    st.markdown(response_data["response"])
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["response"],
                        "analyzed_images": response_data.get("analyzed_images", [])
                    })
    
    @staticmethod
    def _generate_response(query: str) -> Dict[str, Any]:
        """
        Generate response using the full pipeline
        """
        try:
            # Generate query embedding
            embedding_engine = st.session_state.embedding_engine
            query_embedding = embedding_engine.generate_query_embedding(query)
            
            # Retrieve similar documents
            retrieval_engine = st.session_state.retrieval_engine
            retrieved_docs = retrieval_engine.search_similar_documents(query_embedding)
            
            if not retrieved_docs:
                return {"response": "No relevant visual content found. Please upload some documents first."}
            
            # Generate response using Gemini
            generation_engine = st.session_state.generation_engine
            analysis_result = generation_engine.analyze_retrieved_visuals(query, retrieved_docs)
            
            # Prepare analyzed images for display
            analyzed_images = []
            for doc_data, similarity_score in retrieved_docs:
                analyzed_images.append({
                    "image": doc_data["image"],
                    "page_number": doc_data["page_number"],
                    "similarity_score": similarity_score,
                    "visual_elements": analysis_result.get("visual_elements_found", [])
                })
            
            return {
                "response": analysis_result["response"],
                "analyzed_images": analyzed_images
            }
            
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}
    
    @staticmethod
    def _handle_sample_question(question: str):
        """
        Handle sample question selection
        """
        # Add to chat and process
        st.session_state.messages.append({"role": "user", "content": question})
        UIComponents._process_user_query(question)
        st.rerun()
    
    @staticmethod
    def render_visual_results():
        """
        Render visual results and analysis panel
        """
        st.header("🎯 Analysis Results")
        
        # Show recent analysis if available
        if st.session_state.messages:
            latest_assistant_msg = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and "analyzed_images" in msg:
                    latest_assistant_msg = msg
                    break
            
            if latest_assistant_msg:
                st.subheader("📊 Latest Analysis")
                UIComponents._display_analyzed_images(latest_assistant_msg["analyzed_images"])
        
        # Show processing statistics
        if hasattr(st.session_state, 'retrieval_engine'):
            stats = st.session_state.retrieval_engine.get_index_stats()
            
            st.subheader("📈 System Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Pages", stats["total_documents"])
                st.metric("Vector Dimension", stats["vector_dimension"])
            
            with col2:
                st.metric("Similarity Threshold", f"{stats['similarity_threshold']:.2f}")
                status = "🟢 Ready" if stats["is_trained"] else "🔴 No Data"
                st.metric("Status", status)