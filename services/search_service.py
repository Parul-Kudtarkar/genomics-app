# services/search_service.py - Updated for new Pinecone client
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import openai
from pinecone import Pinecone  # New import structure
# Add for caching and connection pooling
from cachetools import LRUCache, cached
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.vector_db import PineconeConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenomicsSearchService:
    """
    Enhanced search service for genomics publications with rich filtering
    """
    
    def __init__(self, config: PineconeConfig = None, openai_api_key: str = None):
        """Initialize search service with configuration"""
        self.config = config or PineconeConfig.from_env()
        self.openai_api_key = openai_api_key
        
        # Initialize Pinecone (updated for v3.0+)
        self.pc = Pinecone(api_key=self.config.api_key)
        self.index = self.pc.Index(self.config.index_name)
        
        # Persistent HTTPX client for connection pooling
        self.httpx_client = httpx.Client(timeout=10.0)
        
        # Initialize OpenAI for embeddings
        self.openai_client = self._init_openai()
        
        # Embedding cache (up to 1000 queries)
        self._embedding_cache = LRUCache(maxsize=1000)
        
        logger.info(f"Search service initialized for index: {self.config.index_name}")
    
    def _init_openai(self):
        """Initialize OpenAI client with persistent HTTPX client"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for generating query embeddings")
        
        try:
            # Use new OpenAI client (v1.10+)
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key, http_client=self.httpx_client)
            # Test connection
            client.models.list()
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception), reraise=True)
    @cached(cache=lambda self: self._embedding_cache)
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query (cached, with retry)"""
        try:
            response = self.openai_client.embeddings.create(
                input=[query_text],
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception), reraise=True)
    def search_similar_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query (with retry)
        
        Args:
            query_text: Natural language search query
            top_k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of matching chunks with metadata and scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_query_embedding(query_text)
            
            # Perform vector search (updated for new Pinecone client)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filters,
                include_metadata=True,
                include_values=False
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': float(match.score),
                    'content': match.metadata.get('text', ''),
                    'title': match.metadata.get('title', 'Unknown'),
                    'source': match.metadata.get('filename', 'Unknown'),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'doc_id': match.metadata.get('doc_id', ''),
                    'metadata': match.metadata
                })
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query_text[:50]}...'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_with_advanced_filters(
        self,
        query_text: str,
        top_k: int = 10,
        journal: str = None,
        author: str = None,
        year_range: Tuple[int, int] = None,
        min_citations: int = None,
        chunk_type: str = None,
        keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with publication metadata filters
        """
        # Build filter dictionary
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
        
        if chunk_type:
            filters["chunk_type"] = {"$eq": chunk_type}
        
        if keywords:
            filters["keywords"] = {"$in": keywords}
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v is not None}
        
        logger.info(f"Searching with filters: {filters}")
        return self.search_similar_chunks(query_text, top_k, filters)
    
    def search_by_document(
        self,
        doc_id: str,
        query_text: str = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific document"""
        filters = {"doc_id": {"$eq": doc_id}}
        
        if query_text:
            return self.search_similar_chunks(query_text, top_k, filters)
        else:
            # Return all chunks from document ordered by chunk_index
            try:
                dummy_query = "research scientific study"  # Neutral query
                results = self.search_similar_chunks(dummy_query, top_k * 3, filters)
                
                # Sort by chunk_index if available
                results.sort(key=lambda x: x.get('chunk_index', 0))
                return results[:top_k]
            except Exception as e:
                logger.error(f"Document search failed: {e}")
                return []
    
    def get_context_chunks(
        self,
        query_text: str,
        max_tokens: int = 4000,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get relevant chunks formatted as context for LLM
        """
        # Search for relevant chunks
        chunks = self.search_similar_chunks(
            query_text=query_text,
            top_k=20,  # Get more initially, then filter by token limit
            filters=filters
        )
        
        if not chunks:
            return [], ""
        
        # Build context string within token limit
        context_parts = []
        selected_chunks = []
        total_chars = 0
        char_limit = max_tokens * 3  # Rough approximation: 1 token ≈ 3 characters
        
        for chunk in chunks:
            content = chunk['content']
            title = chunk['title']
            source = chunk['source']
            
            # Format chunk for context
            chunk_text = f"""
Source: {title} ({source})
Content: {content}
---
"""
            
            # Check if adding this chunk would exceed limit
            if total_chars + len(chunk_text) > char_limit:
                break
            
            context_parts.append(chunk_text)
            selected_chunks.append(chunk)
            total_chars += len(chunk_text)
        
        context_string = "\n".join(context_parts)
        
        logger.info(f"Selected {len(selected_chunks)} chunks for context ({total_chars} characters)")
        return selected_chunks, context_string
    
    def search_high_impact_papers(
        self,
        query_text: str,
        min_citations: int = 20,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for high-impact papers (high citation count)"""
        return self.search_with_advanced_filters(
            query_text=query_text,
            top_k=top_k,
            min_citations=min_citations
        )
    
    def search_recent_papers(
        self,
        query_text: str,
        years_back: int = 3,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for recent papers"""
        from datetime import datetime
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        return self.search_with_advanced_filters(
            query_text=query_text,
            top_k=top_k,
            year_range=(start_year, current_year)
        )
    
    def search_abstracts_only(
        self,
        query_text: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search only in abstract sections"""
        return self.search_with_advanced_filters(
            query_text=query_text,
            top_k=top_k,
            chunk_type="abstract"
        )
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {}

# Convenience function for quick setup
def create_search_service(openai_api_key: str) -> GenomicsSearchService:
    """Create a search service with default configuration"""
    config = PineconeConfig.from_env()
    return GenomicsSearchService(config=config, openai_api_key=openai_api_key)
