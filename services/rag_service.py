#!/usr/bin/env python3
"""
Enhanced Genomics RAG Service
Integrated with existing Pinecone vector store and data ingestion pipelines
"""
import sys
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
import os

# Environment variables loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# LangChain imports for v0.2.x
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.vector_db import PineconeConfig
from services.vector_store import PineconeVectorStore
from services.search_service import GenomicsSearchService

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_service.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structured RAG response with comprehensive metadata"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    num_sources: int
    search_query: str
    model_used: str
    processing_time: float
    filters_used: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGConfig:
    """Configuration for RAG service"""
    # LLM Configuration
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Search Configuration
    default_top_k: int = 5
    max_context_tokens: int = 4000
    chunk_overlap: int = 200
    
    # Prompt Configuration
    system_prompt: str = "You are an expert genomics researcher assistant."
    include_sources: bool = True
    include_metadata: bool = True
    
    # Performance Configuration
    enable_caching: bool = True
    cache_size: int = 1000
    timeout_seconds: int = 30
    
    # Filtering Configuration
    default_filters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        return cls(
            model_name=os.getenv('DEFAULT_LLM_MODEL', 'gpt-4'),
            temperature=float(os.getenv('DEFAULT_TEMPERATURE', '0.1')),
            max_tokens=int(os.getenv('MAX_TOKENS', '4000')),
            default_top_k=int(os.getenv('DEFAULT_TOP_K', '5')),
            max_context_tokens=int(os.getenv('MAX_CONTEXT_TOKENS', '4000')),
            enable_caching=os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            cache_size=int(os.getenv('CACHE_SIZE', '1000')),
            timeout_seconds=int(os.getenv('RAG_TIMEOUT', '30'))
        )

class SimpleGenomicsRetriever:
    """
    Simple retriever that integrates with existing vector store
    Works around LangChain's strict field validation
    """
    
    def __init__(self, vector_store: PineconeVectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        self.filters = None
        self.logger = logging.getLogger(__name__)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            # Use the existing vector store search functionality
            results = self.vector_store.search_similar_chunks(
                query_text=query,
                top_k=self.top_k,
                filters=self.filters
            )
            
            # Convert to LangChain Documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result['content'],
                    metadata={
                        'title': result.get('title', 'Unknown'),
                        'source': result.get('source', 'Unknown'),
                        'score': result.get('score', 0),
                        'chunk_index': result.get('chunk_index', 0),
                        'doc_id': result.get('doc_id', ''),
                        'paper_id': result.get('paper_id', ''),
                        'journal': result.get('journal'),
                        'year': result.get('year'),
                        'authors': result.get('authors', []),
                        'doi': result.get('doi'),
                        'citation_count': result.get('citation_count', 0),
                        'keywords': result.get('keywords', []),
                        'publication_date': result.get('publication_date'),
                        'source_type': result.get('source_type', 'unknown'),
                        **result.get('metadata', {})
                    }
                )
                documents.append(doc)
            
            self.logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents
            
        except Exception as e:
            self.logger.error(f"Document retrieval failed: {e}")
            return []
    
    def set_filters(self, filters: Dict[str, Any]):
        """Set metadata filters for searches"""
        self.filters = filters
    
    def clear_filters(self):
        """Clear all metadata filters"""
        self.filters = None

class GenomicsRAGService:
    """
    Enhanced RAG (Retrieval-Augmented Generation) service for genomics research
    Integrated with existing Pinecone vector store and data ingestion pipelines
    """
    
    def __init__(
        self,
        config: RAGConfig = None,
        vector_store: PineconeVectorStore = None,
        openai_api_key: str = None
    ):
        """
        Initialize RAG service with existing components
        """
        self.config = config or RAGConfig.from_env()
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for RAG service")
        
        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            pinecone_config = PineconeConfig.from_env()
            self.vector_store = PineconeVectorStore(pinecone_config)
        
        # Initialize LLM with correct parameters for LangChain v0.2.x
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds
        )
        
        # Initialize simple retriever
        self.retriever = SimpleGenomicsRetriever(
            vector_store=self.vector_store, 
            top_k=self.config.default_top_k
        )
        
        # Create custom prompt templates
        self.prompt_templates = self._create_prompt_templates()
        
        # Initialize caching if enabled
        self.cache = None
        if self.config.enable_caching:
            from cachetools import LRUCache
            self.cache = LRUCache(maxsize=self.config.cache_size)
        
        logger.info(f"🧬 Genomics RAG Service initialized")
        logger.info(f"   🤖 Model: {self.config.model_name}")
        logger.info(f"   📊 Vector Store: {self.vector_store.config.index_name}")
        logger.info(f"   🔍 Default Top-K: {self.config.default_top_k}")
        logger.info(f"   💾 Caching: {'Enabled' if self.config.enable_caching else 'Disabled'}")
    
    def _create_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Create custom prompt templates for different types of questions"""
        
        # Base genomics research template (Direct Answer)
        base_template = """You are an expert genomics researcher assistant. Use the following research papers and scientific literature to answer the question. 

When answering:
1. Base your response primarily on the provided context
2. Cite specific papers when making claims (use paper titles)
3. If the context doesn't contain enough information, say so clearly
4. Provide scientific detail when appropriate
5. Distinguish between established facts and ongoing research
6. Use proper scientific terminology

Context from research literature:
{context}

Question: {question}

Comprehensive Answer:"""
        
        # Chain of Thought template for complex reasoning
        cot_template = """You are an expert genomics researcher assistant. Use the following research papers to answer the question through careful reasoning.

When answering, follow these steps:
1. First, analyze what the question is asking
2. Identify the key concepts and terms
3. Search through the provided context for relevant information
4. Evaluate the quality and relevance of each source
5. Synthesize the information step by step
6. Draw conclusions based on the evidence
7. Provide your final answer with reasoning

Context from research literature:
{context}

Question: {question}

Let me think through this step by step:

Step 1: Understanding the question
[Analyze what the question is asking and identify key concepts]

Step 2: Identifying relevant information
[What information from the context is relevant to the question]

Step 3: Evaluating the sources
[Assessment of source quality, relevance, and reliability]

Step 4: Synthesizing the information
[How the information fits together and supports different aspects of the answer]

Step 5: Drawing conclusions
[Your reasoning process and logical conclusions]

Final Answer: [Your comprehensive answer with clear reasoning and citations]"""
        
        # Methods-focused template with CoT
        methods_cot_template = """You are an expert genomics researcher assistant specializing in laboratory methods and protocols. 

Use the following research papers to answer questions about experimental methods, protocols, and techniques through systematic analysis.

When answering, follow these steps:
1. Identify the specific methods or techniques being asked about
2. Find relevant methodological information in the context
3. Compare different approaches and protocols
4. Evaluate the advantages and limitations of each method
5. Consider practical implementation details
6. Provide step-by-step reasoning

Context from research literature:
{context}

Question: {question}

Let me analyze the methods step by step:

Step 1: Method Identification
[What specific methods or techniques are relevant to this question]

Step 2: Source Analysis
[What methodological information is available in the provided papers]

Step 3: Comparative Analysis
[How do different methods compare and what are their variations]

Step 4: Practical Considerations
[What are the implementation details, requirements, and limitations]

Step 5: Recommendations
[Based on the evidence, what are the best approaches]

Detailed Methods Answer: [Comprehensive answer with reasoning and citations]"""
        
        # Results-focused template with CoT
        results_cot_template = """You are an expert genomics researcher assistant specializing in research results and findings.

Use the following research papers to answer questions about experimental results, data analysis, and scientific findings through systematic evaluation.

When answering, follow these steps:
1. Identify the specific results or findings being asked about
2. Extract quantitative and qualitative data from the context
3. Evaluate statistical significance and reliability
4. Compare results across different studies
5. Consider implications and limitations
6. Provide evidence-based conclusions

Context from research literature:
{context}

Question: {question}

Let me analyze the results step by step:

Step 1: Result Identification
[What specific results or findings are relevant to this question]

Step 2: Data Extraction
[What quantitative and qualitative data is available in the papers]

Step 3: Statistical Evaluation
[Assessment of significance, reliability, and confidence intervals]

Step 4: Cross-Study Comparison
[How do results compare across different studies and conditions]

Step 5: Interpretation
[What do these results mean and what are their implications]

Results Analysis: [Comprehensive analysis with reasoning and citations]"""
        
        # Comparative analysis template with CoT
        comparison_cot_template = """You are an expert genomics researcher assistant specializing in comparative analysis.

Use the following research papers to compare and contrast different approaches, methods, or findings through systematic evaluation.

When answering, follow these steps:
1. Identify the specific approaches or methods to compare
2. Extract key characteristics of each approach
3. Analyze similarities and differences systematically
4. Evaluate advantages and limitations of each
5. Consider contextual factors and applicability
6. Provide balanced, evidence-based comparison

Context from research literature:
{context}

Question: {question}

Let me compare these approaches step by step:

Step 1: Approach Identification
[What specific approaches, methods, or findings need to be compared]

Step 2: Characteristic Analysis
[What are the key characteristics of each approach]

Step 3: Similarity Assessment
[What are the common features and shared aspects]

Step 4: Difference Analysis
[What are the key differences and distinguishing features]

Step 5: Evaluation
[What are the advantages, limitations, and trade-offs of each approach]

Step 6: Contextual Considerations
[When and where is each approach most appropriate]

Comparative Analysis: [Comprehensive comparison with reasoning and citations]"""
        
        # Methods-focused template (Direct Answer)
        methods_template = """You are an expert genomics researcher assistant specializing in laboratory methods and protocols. 

Use the following research papers to answer questions about experimental methods, protocols, and techniques.

When answering:
1. Focus on practical experimental details
2. Include specific protocols, reagents, and conditions
3. Mention any variations or modifications
4. Cite the source papers for each method
5. Note any limitations or considerations

Context from research literature:
{context}

Question: {question}

Detailed Methods Answer:"""
        
        # Results-focused template (Direct Answer)
        results_template = """You are an expert genomics researcher assistant specializing in research results and findings.

Use the following research papers to answer questions about experimental results, data analysis, and scientific findings.

When answering:
1. Focus on key findings and results
2. Include quantitative data when available
3. Discuss statistical significance
4. Compare results across studies
5. Cite specific papers for each finding

Context from research literature:
{context}

Question: {question}

Results Analysis:"""
        
        # Comparative analysis template (Direct Answer)
        comparison_template = """You are an expert genomics researcher assistant specializing in comparative analysis.

Use the following research papers to compare and contrast different approaches, methods, or findings.

When answering:
1. Clearly identify the approaches being compared
2. Highlight key differences and similarities
3. Discuss advantages and limitations of each
4. Provide evidence from the literature
5. Give balanced, objective analysis

Context from research literature:
{context}

Question: {question}

Comparative Analysis:"""
        
        return {
            'base': PromptTemplate(template=base_template, input_variables=["context", "question"]),
            'cot': PromptTemplate(template=cot_template, input_variables=["context", "question"]),
            'methods': PromptTemplate(template=methods_template, input_variables=["context", "question"]),
            'methods_cot': PromptTemplate(template=methods_cot_template, input_variables=["context", "question"]),
            'results': PromptTemplate(template=results_template, input_variables=["context", "question"]),
            'results_cot': PromptTemplate(template=results_cot_template, input_variables=["context", "question"]),
            'comparison': PromptTemplate(template=comparison_template, input_variables=["context", "question"]),
            'comparison_cot': PromptTemplate(template=comparison_cot_template, input_variables=["context", "question"])
        }
    
    def _get_cached_response(self, query: str, filters: Dict[str, Any] = None) -> Optional[RAGResponse]:
        """Get cached response if available"""
        if not self.cache:
            return None
        
        # Create cache key
        cache_key = self._create_cache_key(query, filters)
        return self.cache.get(cache_key)
    
    def _cache_response(self, query: str, filters: Dict[str, Any], response: RAGResponse):
        """Cache response for future use"""
        if not self.cache:
            return
        
        cache_key = self._create_cache_key(query, filters)
        self.cache[cache_key] = response
    
    def _create_cache_key(self, query: str, filters: Dict[str, Any] = None) -> str:
        """Create unique cache key for query and filters"""
        key_data = {
            'query': query.lower().strip(),
            'filters': filters or {},
            'model': self.config.model_name,
            'top_k': self.retriever.top_k
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def ask_question(
        self,
        question: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        prompt_type: str = 'base',
        include_sources: bool = True
    ) -> RAGResponse:
        """
        Ask a question and get a comprehensive RAG response
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = self._get_cached_response(question, filters)
            if cached_response:
                logger.info(f"📋 Using cached response for query: '{question[:50]}...'")
                return cached_response
            
            # Set retriever parameters
            if top_k is not None:
                self.retriever.top_k = top_k
            else:
                self.retriever.top_k = self.config.default_top_k
            
            if filters:
                self.retriever.set_filters(filters)
            else:
                self.retriever.clear_filters()
            
            # Get relevant documents
            documents = self.retriever.get_relevant_documents(question)
            
            if not documents:
                response = RAGResponse(
                    question=question,
                    answer="I couldn't find any relevant information in the research papers to answer this question.",
                    sources=[],
                    num_sources=0,
                    search_query=question,
                    model_used=self.config.model_name,
                    processing_time=time.time() - start_time,
                    filters_used=filters,
                    confidence_score=0.0
                )
                return response
            
            # Format context from documents
            context_parts = []
            for doc in documents:
                source_info = f"Source: {doc.metadata.get('title', 'Unknown')}"
                if doc.metadata.get('journal'):
                    source_info += f" ({doc.metadata.get('journal')})"
                if doc.metadata.get('year'):
                    source_info += f" ({doc.metadata.get('year')})"
                
                context_parts.append(f"{source_info}\nContent: {doc.page_content}\n---")
            
            context = "\n".join(context_parts)
            
            # Select prompt template
            template = self.prompt_templates.get(prompt_type, self.prompt_templates['base'])
            
            # Create prompt and get answer
            prompt = template.format(context=context, question=question)
            
            # Get response from LLM
            llm_response = self.llm.invoke(prompt)
            answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Extract and format source information
            sources = []
            for doc in documents:
                source_info = {
                    'title': doc.metadata.get('title', 'Unknown'),
                    'source_file': doc.metadata.get('source', 'Unknown'),
                    'relevance_score': doc.metadata.get('score', 0),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'journal': doc.metadata.get('journal'),
                    'year': doc.metadata.get('year'),
                    'authors': doc.metadata.get('authors', []),
                    'doi': doc.metadata.get('doi'),
                    'citation_count': doc.metadata.get('citation_count', 0),
                    'keywords': doc.metadata.get('keywords', []),
                    'publication_date': doc.metadata.get('publication_date'),
                    'source_type': doc.metadata.get('source_type', 'unknown'),
                    'paper_id': doc.metadata.get('paper_id'),
                    'chunk_index': doc.metadata.get('chunk_index', 0)
                }
                sources.append(source_info)
            
            # Sort sources by relevance score
            sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Calculate confidence score based on source quality
            confidence_score = self._calculate_confidence_score(sources, answer)
            
            # Create response
            response = RAGResponse(
                question=question,
                answer=answer,
                sources=sources if include_sources else [],
                num_sources=len(sources),
                search_query=question,
                model_used=self.config.model_name,
                processing_time=time.time() - start_time,
                filters_used=filters,
                confidence_score=confidence_score,
                metadata={
                    'prompt_type': prompt_type,
                    'context_length': len(context),
                    'num_documents': len(documents),
                    'cache_hit': False
                }
            )
            
            # Cache the response
            self._cache_response(question, filters, response)
            
            logger.info(f"✅ Generated answer for question: '{question[:50]}...' using {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return RAGResponse(
                question=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                num_sources=0,
                search_query=question,
                model_used=self.config.model_name,
                processing_time=time.time() - start_time,
                filters_used=filters,
                error=str(e),
                confidence_score=0.0
            )
    
    def _calculate_confidence_score(self, sources: List[Dict[str, Any]], answer: str) -> float:
        """Calculate confidence score based on source quality and answer characteristics"""
        if not sources:
            return 0.0
        
        # Base score from source relevance
        avg_relevance = sum(s['relevance_score'] for s in sources) / len(sources)
        
        # Boost for high-quality sources (high citation count, recent papers)
        quality_boost = 0.0
        for source in sources:
            if source.get('citation_count', 0) > 50:
                quality_boost += 0.1
            if source.get('year', 0) >= 2020:
                quality_boost += 0.05
        
        # Penalty for short answers (might indicate insufficient information)
        length_penalty = 0.0
        if len(answer) < 100:
            length_penalty = 0.2
        elif len(answer) < 200:
            length_penalty = 0.1
        
        # Final score
        confidence = min(1.0, avg_relevance + quality_boost - length_penalty)
        return max(0.0, confidence)
    
    def ask_with_paper_focus(
        self,
        question: str,
        journal: str = None,
        author: str = None,
        year_range: Tuple[int, int] = None,
        min_citations: int = None,
        source_type: str = None,
        top_k: int = None
    ) -> RAGResponse:
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
                    {"crossref_year": {"$gte": start_year, "$lte": end_year}},
                    {"year": {"$gte": start_year, "$lte": end_year}}
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
        
        if source_type:
            filters["source_type"] = {"$eq": source_type}
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v is not None}
        
        return self.ask_question(question, top_k, filters)
    
    def ask_about_methods(self, question: str, top_k: int = None) -> RAGResponse:
        """Ask a question focused on methodology sections"""
        return self.ask_question(
            question, 
            top_k=top_k, 
            prompt_type='methods'
        )
    
    def ask_about_results(self, question: str, top_k: int = None) -> RAGResponse:
        """Ask a question focused on results sections"""
        return self.ask_question(
            question, 
            top_k=top_k, 
            prompt_type='results'
        )
    
    def ask_with_reasoning(self, question: str, top_k: int = None, filters: Optional[Dict[str, Any]] = None) -> RAGResponse:
        """Ask questions with Chain of Thought reasoning"""
        return self.ask_question(
            question=question,
            top_k=top_k,
            filters=filters,
            prompt_type='cot'
        )
    
    def ask_methods_with_reasoning(self, question: str, top_k: int = None) -> RAGResponse:
        """Ask questions about methods with Chain of Thought reasoning"""
        return self.ask_question(
            question=question,
            top_k=top_k,
            prompt_type='methods_cot'
        )
    
    def ask_results_with_reasoning(self, question: str, top_k: int = None) -> RAGResponse:
        """Ask questions about results with Chain of Thought reasoning"""
        return self.ask_question(
            question=question,
            top_k=top_k,
            prompt_type='results_cot'
        )
    
    def compare_with_reasoning(
        self,
        question: str,
        approaches: List[str],
        top_k: int = None
    ) -> RAGResponse:
        """Compare approaches with Chain of Thought reasoning"""
        return self.ask_question(
            question=question,
            top_k=top_k,
            prompt_type='comparison_cot'
        )
    
    def compare_approaches(
        self,
        question: str,
        approaches: List[str],
        top_k: int = None
    ) -> RAGResponse:
        """Compare different research approaches or methods"""
        # Enhance question with approaches
        enhanced_question = f"{question} Specifically compare: {', '.join(approaches)}"
        
        # Add approach keywords to filters
        filters = {"keywords": {"$in": approaches}}
        
        return self.ask_question(
            enhanced_question,
            top_k=top_k,
            filters=filters,
            prompt_type='comparison'
        )
    
    def summarize_recent_research(
        self,
        topic: str,
        years_back: int = 2,
        max_papers: int = 10
    ) -> RAGResponse:
        """Summarize recent research on a topic"""
        from datetime import datetime
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        filters = {
            "$or": [
                {"publication_year": {"$gte": start_year}},
                {"crossref_year": {"$gte": start_year}},
                {"year": {"$gte": start_year}}
            ]
        }
        
        return self.ask_question(
            f"Summarize the recent research trends and findings in {topic}",
            top_k=max_papers,
            filters=filters
        )
    
    def search_by_document(
        self,
        doc_id: str,
        question: str = None,
        top_k: int = None
    ) -> RAGResponse:
        """Search within a specific document"""
        filters = {"doc_id": {"$eq": doc_id}}
        
        if question:
            return self.ask_question(question, top_k, filters)
        else:
            # Return document summary
            return self.ask_question(
                f"Summarize the key findings and content of this document",
                top_k=top_k,
                filters=filters
            )
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_index_statistics()
            
            # Get cache stats
            cache_stats = {
                'enabled': self.config.enable_caching,
                'size': len(self.cache) if self.cache else 0,
                'max_size': self.config.cache_size if self.cache else 0
            }
            
            return {
                'vector_store': vector_stats,
                'cache': cache_stats,
                'config': {
                    'model': self.config.model_name,
                    'temperature': self.config.temperature,
                    'default_top_k': self.config.default_top_k,
                    'max_context_tokens': self.config.max_context_tokens
                },
                'service_info': {
                    'initialized_at': datetime.now().isoformat(),
                    'prompt_templates': list(self.prompt_templates.keys())
                }
            }
        except Exception as e:
            logger.error(f"Failed to get service statistics: {e}")
            return {'error': str(e)}

# Convenience function for quick setup
def create_rag_service(
    openai_api_key: str = None,
    config: RAGConfig = None,
    vector_store: PineconeVectorStore = None
) -> GenomicsRAGService:
    """Create a RAG service with default configuration"""
    return GenomicsRAGService(
        config=config,
        vector_store=vector_store,
        openai_api_key=openai_api_key
    )

# Test function
def test_rag_service():
    """Test the RAG service with a sample question"""
    try:
        print("🧬 Testing Genomics RAG Service")
        print("=" * 50)
        
        # Create RAG service
        rag = create_rag_service()
        
        # Test basic question
        print("🔍 Testing basic question...")
        response = rag.ask_question("What is CRISPR gene editing?", top_k=3)
        
        print(f"✅ Test successful!")
        print(f"   Answer length: {len(response.answer)} characters")
        print(f"   Sources found: {response.num_sources}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   Confidence score: {response.confidence_score:.2f}")
        print(f"   Preview: {response.answer[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_rag_service()
