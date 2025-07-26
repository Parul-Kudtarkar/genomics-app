#enhanced_search.py 
#!/usr/bin/env python3
"""
Enhanced search functions for your vector store
"""
import os
import openai
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedVectorSearch:
    """Enhanced search functions for vector store"""
    
    def __init__(self, pinecone_api_key: str = None, openai_api_key: str = None, index_name: str = None):
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        # Initialize clients
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        self.openai_client = self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            # Test connection
            client.models.list()
            return client
        except:
            # Legacy fallback
            openai.api_key = self.openai_api_key
            return None
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for query text"""
        try:
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    input=[query_text],
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            else:
                response = openai.Embedding.create(
                    input=[query_text],
                    model="text-embedding-ada-002"
                )
                return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def search_with_filters(self, 
                          query_vector: List[float],
                          author: str = None,
                          journal: str = None,
                          year_range: tuple = None,
                          keywords: List[str] = None,
                          institution: str = None,
                          chunk_type: str = None,
                          min_citations: int = None,
                          publisher: str = None,
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search with multiple filters"""
        
        # Build metadata filter
        filter_conditions = {}
        
        if author:
            filter_conditions["authors"] = {"$in": [author]}
        
        if journal:
            # Search in both journal fields
            filter_conditions["$or"] = [
                {"journal": {"$eq": journal}},
                {"crossref_journal": {"$eq": journal}}
            ]
        
        if year_range:
            start_year, end_year = year_range
            # Search in both year fields
            year_filter = {
                "$or": [
                    {"publication_year": {"$gte": start_year, "$lte": end_year}},
                    {"crossref_year": {"$gte": start_year, "$lte": end_year}}
                ]
            }
            if filter_conditions.get("$or"):
                filter_conditions["$and"] = [
                    {"$or": filter_conditions.pop("$or")},
                    year_filter
                ]
            else:
                filter_conditions.update(year_filter)
        
        if keywords:
            filter_conditions["keywords"] = {"$in": keywords}
        
        if institution:
            filter_conditions["institutions"] = {"$in": [institution]}
        
        if chunk_type:
            filter_conditions["chunk_type"] = {"$eq": chunk_type}
        
        if min_citations:
            filter_conditions["citation_count"] = {"$gte": min_citations}
        
        if publisher:
            filter_conditions["publisher"] = {"$eq": publisher}
        
        # Perform search
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_conditions if filter_conditions else None,
                include_metadata=True,
                include_values=False
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text(self, query_text: str, **kwargs) -> List[Dict[str, Any]]:
        """Search using text query (generates embedding automatically)"""
        query_vector = self.generate_query_embedding(query_text)
        if not query_vector:
            return []
        
        return self.search_with_filters(query_vector=query_vector, **kwargs)
    
    def find_similar_papers_by_author(self, author_name: str, query_text: str, top_k: int = 5):
        """Find papers similar to query by specific author"""
        return self.search_by_text(
            query_text=query_text,
            author=author_name,
            top_k=top_k
        )
    
    def get_papers_by_year(self, year: int, top_k: int = 20):
        """Get papers from a specific year"""
        # Use a neutral query for year-based search
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            year_range=(year, year),
            top_k=top_k
        )
    
    def get_high_impact_papers(self, min_citations: int = 50, top_k: int = 10):
        """Get highly cited papers"""
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            min_citations=min_citations,
            top_k=top_k
        )
    
    def search_abstracts_only(self, query_text: str, top_k: int = 10):
        """Search only in abstract chunks"""
        return self.search_by_text(
            query_text=query_text,
            chunk_type="abstract",
            top_k=top_k
        )
    
    def get_papers_by_institution(self, institution: str, top_k: int = 15):
        """Get papers from specific institution"""
        dummy_query = "research scientific study"
        return self.search_by_text(
            query_text=dummy_query,
            institution=institution,
            top_k=top_k
        )
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics from your vector store"""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.total_vector_count,
                "namespace_counts": stats.namespaces,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
            
        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            return {}
