# services/rag_service.py - Fixed version with simple retriever
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os

# LangChain imports for v0.2.x
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from services.search_service import GenomicsSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGenomicsRetriever:
    """
    Simple retriever that doesn't inherit from BaseRetriever
    Works around LangChain's strict field validation
    """
    
    def __init__(self, search_service: GenomicsSearchService, top_k: int = 5):
        self.search_service = search_service
        self.top_k = top_k
        self.filters = None
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            # Search for relevant chunks
            chunks = self.search_service.search_similar_chunks(
                query_text=query,
                top_k=self.top_k,
                filters=self.filters
            )
            
            # Convert to LangChain Documents
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'title': chunk['title'],
                        'source': chunk['source'],
                        'score': chunk['score'],
                        'chunk_index': chunk['chunk_index'],
                        'doc_id': chunk['doc_id'],
                        **chunk['metadata']  # Include all original metadata
                    }
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def set_filters(self, filters: Dict[str, Any]):
        """Set metadata filters for searches"""
        self.filters = filters
    
    def clear_filters(self):
        """Clear all metadata filters"""
        self.filters = None

class GenomicsRAGService:
    """
    RAG (Retrieval-Augmented Generation) service for genomics questions using LangChain
    """
    
    def __init__(
        self,
        search_service: GenomicsSearchService,
        openai_api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.1
    ):
        """
        Initialize RAG service
        """
        self.search_service = search_service
        self.openai_api_key = openai_api_key
        
        # Initialize LLM with correct parameters for LangChain v0.2.x
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature
        )
        
        # Initialize simple retriever
        self.retriever = SimpleGenomicsRetriever(search_service=search_service, top_k=5)
        
        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"RAG service initialized with model: {model_name}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for genomics questions"""
        template = """You are an expert genomics researcher assistant. Use the following research papers and scientific literature to answer the question. 

When answering:
1. Base your response primarily on the provided context
2. Cite specific papers when making claims (use paper titles)
3. If the context doesn't contain enough information, say so clearly
4. Provide scientific detail when appropriate
5. Distinguish between established facts and ongoing research

Context from research literature:
{context}

Question: {question}

Comprehensive Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception), reraise=True)
    def _invoke_llm(self, prompt):
        return self.llm.invoke(prompt)
    
    def ask_question(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources - simplified approach
        """
        try:
            # Set retriever parameters
            self.retriever.top_k = top_k
            if filters:
                self.retriever.set_filters(filters)
            else:
                self.retriever.clear_filters()
            
            # Get relevant documents
            documents = self.retriever.get_relevant_documents(question)
            
            if not documents:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information in the research papers to answer this question.",
                    'sources': [],
                    'num_sources': 0
                }
            
            # Format context from documents
            context_parts = []
            for doc in documents:
                context_parts.append(f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}\n---")
            
            context = "\n".join(context_parts)
            
            # Create prompt and get answer
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Get response from LLM
            response = self._invoke_llm(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Extract and format source information
            sources = []
            for doc in documents:
                source_info = {
                    'title': doc.metadata.get('title', 'Unknown'),
                    'source_file': doc.metadata.get('source', 'Unknown'),
                    'relevance_score': doc.metadata.get('score', 0),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'journal': doc.metadata.get('journal') or doc.metadata.get('crossref_journal'),
                    'year': doc.metadata.get('publication_year') or doc.metadata.get('crossref_year'),
                    'authors': doc.metadata.get('authors', []),
                    'doi': doc.metadata.get('doi'),
                    'citation_count': doc.metadata.get('citation_count', 0)
                }
                sources.append(source_info)
            
            # Sort sources by relevance score
            sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            response = {
                'question': question,
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources),
                'filters_used': filters,
                'search_query': question
            }
            
            logger.info(f"Generated answer for question: '{question[:50]}...' using {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'question': question,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'error': str(e)
            }
    
    def ask_with_paper_focus(
        self,
        question: str,
        journal: str = None,
        author: str = None,
        year_range: Tuple[int, int] = None,
        min_citations: int = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Ask a question with focus on specific types of papers"""
        # Build filters
        filters = {}
        
        if journal:
            filters["$or"] = [
                {"journal": {"$eq": journal}},
                {"crossref_journal": {"$eq": journal}}
            ]
        
        if author:
            filters["authors"] = {"$in": [author]}
        
        if year_range:
            start_year, end_year = year_range
            year_filter = {
                "$or": [
                    {"publication_year": {"$gte": start_year, "$lte": end_year}},
                    {"crossref_year": {"$gte": start_year, "$lte": end_year}}
                ]
            }
            if filters.get("$or"):
                filters["$and"] = [
                    {"$or": filters.pop("$or")},
                    year_filter
                ]
            else:
                filters.update(year_filter)
        
        if min_citations:
            filters["citation_count"] = {"$gte": min_citations}
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v is not None}
        
        return self.ask_question(question, top_k, filters)
    
    def ask_about_methods(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Ask a question focused on methodology sections"""
        filters = {"chunk_type": {"$eq": "methods"}}
        return self.ask_question(question, top_k, filters)
    
    def ask_about_results(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Ask a question focused on results sections"""
        filters = {"chunk_type": {"$eq": "results"}}
        return self.ask_question(question, top_k, filters)
    
    def compare_approaches(
        self,
        question: str,
        approaches: List[str],
        top_k: int = 8
    ) -> Dict[str, Any]:
        """Compare different research approaches or methods"""
        try:
            all_sources = []
            approach_results = {}
            
            # Search for each approach
            for approach in approaches:
                combined_query = f"{question} {approach}"
                filters = {"keywords": {"$in": [approach]}}
                
                chunks = self.search_service.search_similar_chunks(
                    query_text=combined_query,
                    top_k=top_k // len(approaches) + 1,
                    filters=filters
                )
                
                approach_results[approach] = chunks
                all_sources.extend(chunks)
            
            # Create a comprehensive comparison prompt
            comparison_template = """You are comparing different research approaches in genomics. 

Based on the provided research literature, compare and contrast the following approaches: {approaches}

Research Context:
{context}

Question: {question}

Provide a detailed comparison that:
1. Explains each approach based on the literature
2. Highlights key differences and similarities  
3. Discusses advantages and limitations of each
4. Cites specific papers for each approach
5. Provides recommendations if appropriate

Comparative Analysis:"""

            # Format context from all sources
            context_parts = []
            for source in all_sources[:top_k]:
                context_parts.append(f"Source: {source['title']}\nContent: {source['content']}\n---")
            
            context = "\n".join(context_parts)
            
            # Generate comparison using invoke method
            comparison_prompt = comparison_template.format(
                approaches=", ".join(approaches),
                context=context,
                question=question
            )
            
            response = self.llm.invoke(comparison_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                'question': question,
                'approaches_compared': approaches,
                'answer': answer,
                'sources_by_approach': approach_results,
                'total_sources': len(all_sources),
                'comparison_type': 'multi_approach'
            }
            
        except Exception as e:
            logger.error(f"Approach comparison failed: {e}")
            return {
                'question': question,
                'answer': f"Error generating comparison: {str(e)}",
                'error': str(e)
            }
    
    def summarize_recent_research(
        self,
        topic: str,
        years_back: int = 2,
        max_papers: int = 10
    ) -> Dict[str, Any]:
        """Summarize recent research on a topic"""
        from datetime import datetime
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        filters = {
            "$or": [
                {"publication_year": {"$gte": start_year}},
                {"crossref_year": {"$gte": start_year}}
            ]
        }
        
        return self.ask_question(
            f"Summarize the recent research trends and findings in {topic}",
            top_k=max_papers,
            filters=filters
        )

# Convenience function for quick setup
def create_rag_service(
    openai_api_key: str,
    model_name: str = "gpt-4",
    temperature: float = 0.1
) -> GenomicsRAGService:
    """Create a RAG service with default configuration"""
    search_service = GenomicsSearchService(openai_api_key=openai_api_key)
    return GenomicsRAGService(
        search_service=search_service,
        openai_api_key=openai_api_key,
        model_name=model_name,
        temperature=temperature
    )
