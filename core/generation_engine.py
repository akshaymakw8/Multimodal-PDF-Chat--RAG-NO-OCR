# ===========================================================================
# GEMINI GENERATION ENGINE - core/generation_engine.py
# ===========================================================================

import google.generativeai as genai
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Tuple
import streamlit as st

class GeminiGenerationEngine:
    """
    Handles visual analysis and response generation using Google Gemini 2.5 Flash
    Analyzes retrieved visual content to generate precise answers
    """
    
    def __init__(self):
        """
        Initialize Gemini client and configuration
        """
        from config.settings import AppConfig
        self.config = AppConfig()
        
        try:
            genai.configure(api_key=self.config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {str(e)}")
            self.model = None
    
    # ---------------------------------------------------------------------------
    # Visual Analysis and Generation
    # ---------------------------------------------------------------------------
    
    def analyze_retrieved_visuals(
        self, 
        query: str, 
        retrieved_documents: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """
        Analyze retrieved visual documents and generate comprehensive response
        
        Args:
            query: User's original question
            retrieved_documents: List of (document_data, similarity_score) tuples
            
        Returns:
            Dictionary containing analysis results and generated response
        """
        if not self.model:
            return {"error": "Gemini model not initialized"}
        
        if not retrieved_documents:
            return {"response": "No relevant visual content found for your question."}
        
        try:
            # Prepare visual content for analysis
            visual_context = self._prepare_visual_context(retrieved_documents)
            
            # Generate comprehensive analysis
            analysis_result = self._generate_visual_analysis(query, visual_context)
            
            return {
                "response": analysis_result["response"],
                "analyzed_pages": analysis_result["analyzed_pages"],
                "confidence_scores": analysis_result["confidence_scores"],
                "visual_elements_found": analysis_result["visual_elements"],
                "page_references": analysis_result["page_references"]
            }
            
        except Exception as e:
            st.error(f"Error during visual analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _prepare_visual_context(
        self, 
        retrieved_documents: List[Tuple[Dict[str, Any], float]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare visual context from retrieved documents for Gemini analysis
        
        Args:
            retrieved_documents: Retrieved document data with similarity scores
            
        Returns:
            List of prepared visual context dictionaries
        """
        visual_context = []
        
        for doc_data, similarity_score in retrieved_documents:
            context_item = {
                "image": doc_data["image"],
                "page_number": doc_data["page_number"],
                "similarity_score": similarity_score,
                "metadata": doc_data["metadata"],
                "doc_id": doc_data["id"]
            }
            visual_context.append(context_item)
        
        return visual_context
    
    def _generate_visual_analysis(
        self, 
        query: str, 
        visual_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive visual analysis using Gemini
        
        Args:
            query: User's question
            visual_context: Prepared visual context
            
        Returns:
            Dictionary containing detailed analysis results
        """
        # ---------------------------------------------------------------------------
        # Single Image Analysis (Most Relevant)
        # ---------------------------------------------------------------------------
        if len(visual_context) == 1:
            return self._analyze_single_image(query, visual_context[0])
        
        # ---------------------------------------------------------------------------
        # Multi-Image Comparative Analysis
        # ---------------------------------------------------------------------------
        return self._analyze_multiple_images(query, visual_context)
    
    def _analyze_single_image(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single most relevant image
        """
        try:
            # Construct detailed prompt for single image analysis
            prompt = self._build_single_image_prompt(query, context)
            
            # Generate response using Gemini's vision capabilities
            response = self.model.generate_content([prompt, context["image"]])
            
            return {
                "response": response.text,
                "analyzed_pages": [context["page_number"]],
                "confidence_scores": [context["similarity_score"]],
                "visual_elements": self._extract_visual_elements(response.text),
                "page_references": [f"Page {context['page_number']}"]
            }
            
        except Exception as e:
            return {
                "response": f"Error analyzing image: {str(e)}",
                "analyzed_pages": [],
                "confidence_scores": [],
                "visual_elements": [],
                "page_references": []
            }
    
    def _analyze_multiple_images(
        self, 
        query: str, 
        visual_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze multiple images for comparative insights
        """
        try:
            # Analyze each image individually first
            individual_analyses = []
            for context in visual_context[:3]:  # Limit to top 3 for efficiency
                individual_result = self._analyze_single_image(query, context)
                individual_analyses.append({
                    "page": context["page_number"],
                    "analysis": individual_result["response"],
                    "score": context["similarity_score"]
                })
            
            # Generate comparative summary
            comparative_prompt = self._build_comparative_prompt(query, individual_analyses)
            summary_response = self.model.generate_content(comparative_prompt)
            
            return {
                "response": summary_response.text,
                "analyzed_pages": [ctx["page_number"] for ctx in visual_context],
                "confidence_scores": [ctx["similarity_score"] for ctx in visual_context],
                "visual_elements": self._extract_visual_elements(summary_response.text),
                "page_references": [f"Page {ctx['page_number']}" for ctx in visual_context],
                "individual_analyses": individual_analyses
            }
            
        except Exception as e:
            return {
                "response": f"Error in comparative analysis: {str(e)}",
                "analyzed_pages": [],
                "confidence_scores": [],
                "visual_elements": [],
                "page_references": []
            }
    
    # ---------------------------------------------------------------------------
    # Prompt Engineering for Visual Analysis
    # ---------------------------------------------------------------------------
    
    def _build_single_image_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Build detailed prompt for single image analysis
        """
        return f"""
        You are an expert visual analyst specializing in document understanding.
        
        **User Question:** {query}
        
        **Context:** This image is from page {context['page_number']} of a document.
        **Similarity Score:** {context['similarity_score']:.3f} (how relevant this page is to the question)
        
        **Instructions:**
        1. Carefully examine ALL visual elements in this image
        2. Look for: charts, graphs, tables, diagrams, text, numbers, trends, patterns
        3. Focus specifically on elements that answer the user's question
        4. If you find relevant information, explain it clearly and precisely
        5. If the image contains charts/graphs, describe the data, trends, and key insights
        6. Mention specific page number in your response
        7. Be concise but comprehensive
        
        **Response Format:**
        - Start with: "Based on the visual analysis of Page {context['page_number']}:"
        - Provide clear, factual answers
        - Include specific data points, numbers, or trends if visible
        - End with confidence level about the answer
        
        Please analyze this image and answer the user's question.
        """
    
    def _build_comparative_prompt(self, query: str, individual_analyses: List[Dict[str, Any]]) -> str:
        """
        Build prompt for comparative analysis across multiple images
        """
        analyses_text = "\n\n".join([
            f"**Page {analysis['page']} (Score: {analysis['score']:.3f}):**\n{analysis['analysis']}"
            for analysis in individual_analyses
        ])
        
        return f"""
        You are synthesizing insights from multiple document pages to answer a user's question.
        
        **User Question:** {query}
        
        **Individual Page Analyses:**
        {analyses_text}
        
        **Instructions:**
        1. Synthesize information from all analyzed pages
        2. Identify common themes, contradictions, or complementary information
        3. Provide a comprehensive answer that leverages insights from multiple pages
        4. Reference specific pages when mentioning data points
        5. If pages show different aspects of the same topic, explain the relationships
        6. Highlight the most relevant findings
        
        **Response Format:**
        - Start with: "Based on analysis across multiple pages:"
        - Provide synthesized insights
        - Reference specific pages: "As shown on Page X..." 
        - End with overall conclusion and confidence assessment
        
        Please provide a comprehensive answer based on all the visual evidence.
        """
    
    def _extract_visual_elements(self, response_text: str) -> List[str]:
        """
        Extract mentioned visual elements from the generated response
        """
        visual_keywords = [
            "chart", "graph", "table", "diagram", "figure", "image", 
            "plot", "histogram", "bar chart", "pie chart", "line graph",
            "scatter plot", "flowchart", "timeline", "map", "infographic"
        ]
        
        found_elements = []
        response_lower = response_text.lower()
        
        for keyword in visual_keywords:
            if keyword in response_lower:
                found_elements.append(keyword)
        
        return list(set(found_elements))  # Remove duplicates
    
    # ---------------------------------------------------------------------------
    # Specialized Analysis Methods
    # ---------------------------------------------------------------------------
    
    def analyze_financial_charts(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Specialized analysis for financial charts and data
        """
        financial_prompt = f"""
        You are a financial analyst examining a document page.
        
        **Question:** {query}
        **Page:** {context['page_number']}
        
        **Focus Areas:**
        - Revenue, profit, loss figures
        - Growth rates and percentages
        - Time series data and trends
        - Comparative metrics
        - Key performance indicators (KPIs)
        - Market data and statistics
        
        **Instructions:**
        1. Identify all numerical data visible
        2. Explain trends, growth/decline patterns
        3. Highlight significant financial metrics
        4. Provide context for the numbers (time period, currency, etc.)
        5. Answer the specific financial question asked
        
        Analyze this financial document and provide detailed insights.
        """
        
        try:
            response = self.model.generate_content([financial_prompt, context["image"]])
            return {
                "response": response.text,
                "analysis_type": "financial",
                "page_number": context["page_number"]
            }
        except Exception as e:
            return {"error": f"Financial analysis failed: {str(e)}"}
    
    def analyze_technical_diagrams(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Specialized analysis for technical diagrams and schematics
        """
        technical_prompt = f"""
        You are a technical expert analyzing engineering or technical documentation.
        
        **Question:** {query}
        **Page:** {context['page_number']}
        
        **Focus Areas:**
        - System architecture and components
        - Process flows and workflows
        - Technical specifications
        - Relationships between elements
        - Labels, annotations, and callouts
        - Performance metrics or measurements
        
        **Instructions:**
        1. Identify all technical components visible
        2. Explain system relationships and flows
        3. Note any specifications, measurements, or parameters
        4. Answer the technical question with precision
        5. Use appropriate technical terminology
        
        Analyze this technical diagram and provide expert insights.
        """
        
        try:
            response = self.model.generate_content([technical_prompt, context["image"]])
            return {
                "response": response.text,
                "analysis_type": "technical",
                "page_number": context["page_number"]
            }
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    # ---------------------------------------------------------------------------
    # Response Enhancement and Validation
    # ---------------------------------------------------------------------------
    
    def enhance_response_with_context(
        self, 
        base_response: str, 
        query: str, 
        visual_context: List[Dict[str, Any]]
    ) -> str:
        """
        Enhance the base response with additional context and validation
        """
        try:
            enhancement_prompt = f"""
            Please enhance this analysis response with additional context and validation.
            
            **Original Question:** {query}
            **Current Response:** {base_response}
            **Available Pages:** {[ctx['page_number'] for ctx in visual_context]}
            
            **Enhancement Instructions:**
            1. Add confidence indicators for key statements
            2. Suggest follow-up questions if relevant
            3. Mention if additional pages might contain related information
            4. Provide context about limitations of the analysis
            5. Ensure the response directly addresses the user's question
            
            Provide an enhanced, more complete response.
            """
            
            enhanced = self.model.generate_content(enhancement_prompt)
            return enhanced.text
            
        except Exception as e:
            st.warning(f"Could not enhance response: {str(e)}")
            return base_response
    
    def validate_response_accuracy(self, response: str, visual_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the accuracy and completeness of the generated response
        """
        validation_metrics = {
            "confidence_score": 0.0,
            "completeness": "partial",
            "accuracy_indicators": [],
            "potential_gaps": []
        }
        
        try:
            # Simple heuristic validation
            if len(response) > 100:
                validation_metrics["completeness"] = "comprehensive"
            elif len(response) > 50:
                validation_metrics["completeness"] = "adequate"
            
            # Check for specific data mentions
            if any(char.isdigit() for char in response):
                validation_metrics["accuracy_indicators"].append("contains_numerical_data")
            
            if len(visual_context) > 1 and "page" in response.lower():
                validation_metrics["accuracy_indicators"].append("references_multiple_pages")
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(ctx["similarity_score"] for ctx in visual_context) / len(visual_context)
            validation_metrics["confidence_score"] = avg_similarity
            
        except Exception as e:
            st.warning(f"Validation failed: {str(e)}")
        
        return validation_metrics