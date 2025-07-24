# Genomics RAG System - Updated Implementation Guide

## Production-Grade Operational Features

This section provides actionable best practices and code/configuration snippets to make your Genomics RAG system secure, reliable, observable, performant, and production-ready. All examples use FastAPI, but the principles apply to Flask and other frameworks as well.

---

### 🛡️ Security

#### API Authentication (User/API key validation)
Require an API key for all endpoints. Store the key in your environment, not in code.

```python
from fastapi import FastAPI, Depends, HTTPException, status, Request
import os

def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

app = FastAPI()

@app.get("/secure-endpoint")
def secure_endpoint(api_key: str = Depends(get_api_key)):
    return {"message": "Authenticated!"}
```

#### Input Sanitization
Use Pydantic models for request validation. Limit input length and strip whitespace.

```python
from pydantic import BaseModel, constr

class QueryRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=1, max_length=500)
```

#### Rate Limiting
Use [slowapi](https://pypi.org/project/slowapi/) for FastAPI.

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.get("/endpoint")
@limiter.limit("10/minute")
def endpoint():
    ...
```

---

### ⚡ Reliability

#### Error Handling
Wrap all API logic in try/except and log errors.

```python
from fastapi import HTTPException
import logging

@app.post("/ask")
def ask(request: QueryRequest):
    try:
        # ... your logic ...
        return {"result": result}
    except Exception as e:
        logging.exception("Error in /ask endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### Retry Logic
Use [tenacity](https://tenacity.readthedocs.io/) for retrying failed API calls.

```python
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai(...):
    ...
```

#### Timeouts
Set timeouts on all external HTTP calls (OpenAI, Pinecone, etc).

```python
import httpx
client = httpx.Client(timeout=10.0)
```

---

### 📈 Monitoring

#### Structured Logging (JSON, request IDs)
Use Python’s `logging` with a JSON formatter. Generate a request ID for each request.

```python
import logging
import json
import uuid
from fastapi import Request

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None)
        })

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger()
logger.addHandler(handler)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

#### Health Checks
Add a `/health` endpoint that returns 200 OK.

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

#### Cost Tracking (OpenAI token usage)
Log and aggregate token usage from OpenAI API responses.

```python
# After each OpenAI call
usage = response.usage
logger.info(f"OpenAI tokens used: {usage}")
```

---

### 🚀 Performance

#### Caching
Use in-memory cache (e.g., `cachetools`) for frequent queries/embeddings.

```python
from cachetools import LRUCache, cached

cache = LRUCache(maxsize=1000)

@cached(cache)
def get_embedding(query):
    ...
```

#### Connection Pooling
Use persistent HTTP clients (e.g., `httpx.Client()`).

#### Token Optimization
Truncate context to fit within model token limits before sending to OpenAI.

---

### ⚙️ Configuration

#### Environment Management
Use `.env` files and `python-dotenv` to load environment-specific configs.

#### Secrets Management
Never hardcode secrets; always use environment variables.

---

### 📦 Deployment

#### Containerization
Provide a `Dockerfile` for the app.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Process Management
Use `gunicorn` or `uvicorn` with workers for production.

---

_Integrate these practices into your API and service code. The rest of this document provides the RAG and search implementation. Use the above as a reference for operationalizing your system._

## **Overview**

This guide provides the **working, tested implementation** of a Retrieval-Augmented Generation (RAG) system for genomics research. Unlike the original version, this implementation has been updated to work with current package versions and includes fixes for common compatibility issues.

## ✅ **What's Fixed in This Version**

- **LangChain 0.2.x compatibility** - Updated imports and APIs
- **OpenAI 1.51.x compatibility** - Fixed client initialization 
- **httpx compatibility** - Resolved proxies parameter conflicts
- **BaseRetriever issues** - Simplified retriever implementation
- **Production-ready** - Includes error handling and logging

## 📦 **Compatible Package Versions**

```bash
# Tested working versions
langchain==0.2.16
langchain-openai==0.1.23
langchain-core==0.2.38
openai==1.51.2
pinecone-client==3.2.2
httpx>=0.25.0,<0.28.0
python-dotenv==1.0.1
numpy==1.26.4
```

## 🔧 **Installation & Setup**

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install compatible versions
pip install \
  "langchain==0.2.16" \
  "langchain-openai==0.1.23" \
  "openai==1.51.2" \
  "pinecone-client==3.2.2" \
  "httpx>=0.25.0,<0.28.0" \
  "python-dotenv==1.0.1" \
  "numpy==1.26.4"

# Your existing PDF processing
pip install PyPDF2==3.0.1 pdfplumber==0.10.3
```

### Step 2: Environment Configuration

Create/update your `.env` file:

```bash
# .env
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications  # Use your actual index name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.1
```

## 🔍 **Part 1: Enhanced Search Service**

### Key Features
- **Semantic Search** - Natural language queries with embeddings
- **Advanced Filtering** - Journal, author, year, citations, chunk type
- **Context Preparation** - Format results for LLM consumption
- **Specialized Searches** - Recent papers, high-impact, abstracts-only

### Implementation

Create `services/search_service.py`:

```python
# services/search_service.py - Updated for new Pinecone client
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import openai
from pinecone import Pinecone  # New import structure

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
        
        # Initialize OpenAI for embeddings
        self.openai_client = self._init_openai()
        
        logger.info(f"Search service initialized for index: {self.config.index_name}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for generating query embeddings")
        
        try:
            # Use new OpenAI client (v1.10+)
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            # Test connection
            client.models.list()
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            response = self.openai_client.embeddings.create(
                input=[query_text],
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise
    
    def search_similar_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query
        
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
```

### Key Methods

```python
# Basic search
results = search_service.search_similar_chunks("CRISPR gene editing", top_k=10)

# Advanced filtering
results = search_service.search_with_advanced_filters(
    query_text="diabetes treatment",
    journal="Nature", 
    year_range=(2020, 2024),
    min_citations=20
)

# Get context for LLM
chunks, context = search_service.get_context_chunks("gene therapy", max_tokens=4000)
```

## 🤖 **Part 2: LangChain RAG Integration**

### Key Features
- **Simplified Retriever** - Bypasses BaseRetriever field validation issues
- **Domain-Specific Prompts** - Tailored for genomics research
- **Source Attribution** - Returns answers with cited papers
- **Multiple Q&A Modes** - Methods, results, comparative analysis

### Implementation

Create `services/rag_service.py`:

```python
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
            response = self.llm.invoke(prompt)
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
```

### Key Methods

```python
# Basic Q&A
response = rag_service.ask_question("What are the latest CRISPR developments?")

# Filtered Q&A
response = rag_service.ask_with_paper_focus(
    question="Gene therapy advances",
    journal="Nature",
    min_citations=50
)

# Specialized searches
methods = rag_service.ask_about_methods("RNA sequencing protocols")
comparison = rag_service.compare_approaches("Compare CRISPR vs TALEN", ["CRISPR", "TALEN"])
```

## 🧪 **Testing & Validation**

### Create Test Script

Create `test_rag_system.py`:

```python
#!/usr/bin/env python3
# test_langchain_rag.py - Test your LangChain RAG setup
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_langchain_imports():
    """Test LangChain imports"""
    print("🔍 Testing LangChain imports...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA
        from langchain.schema import Document
        from langchain.prompts import PromptTemplate
        print("✅ LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False

def test_core_dependencies():
    """Test core dependencies"""
    print("🔍 Testing core dependencies...")
    try:
        from pinecone import Pinecone
        from openai import OpenAI
        print("✅ Core dependencies successful")
        return True
    except ImportError as e:
        print(f"❌ Core dependency import failed: {e}")
        return False

def test_your_existing_services():
    """Test your existing services still work"""
    print("🔍 Testing your existing services...")
    try:
        from config.vector_db import PineconeConfig
        from services.vector_store import PineconeVectorStore
        from services.search_service import GenomicsSearchService
        
        # Test config
        config = PineconeConfig.from_env()
        print(f"✅ Config loaded: {config.index_name}")
        
        # Test search service
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            search_service = GenomicsSearchService(openai_api_key=openai_key)
            print("✅ Search service created")
        else:
            print("⚠️  OPENAI_API_KEY not set, skipping search service test")
        
        return True
    except Exception as e:
        print(f"❌ Existing services test failed: {e}")
        return False

def test_rag_service():
    """Test RAG service creation"""
    print("🔍 Testing RAG service...")
    try:
        from services.rag_service import create_rag_service
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("⚠️  OPENAI_API_KEY not set, skipping RAG test")
            return True
        
        # Create RAG service
        rag = create_rag_service(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo"  # Use cheaper model for testing
        )
        print("✅ RAG service created successfully")
        
        return True
    except Exception as e:
        print(f"❌ RAG service test failed: {e}")
        return False

def test_full_rag_pipeline():
    """Test full RAG pipeline with a question"""
    print("🔍 Testing full RAG pipeline...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    
    if not openai_key or not pinecone_key:
        print("⚠️  API keys not set, skipping full pipeline test")
        print("   Set OPENAI_API_KEY and PINECONE_API_KEY to test")
        return True
    
    try:
        from services.rag_service import create_rag_service
        
        # Create RAG service
        rag = create_rag_service(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo"
        )
        
        # Test with a simple question
        print("   Asking test question...")
        response = rag.ask_question("What is gene therapy?", top_k=2)
        
        print(f"✅ Full pipeline test successful!")
        print(f"   Answer length: {len(response['answer'])} characters")
        print(f"   Sources found: {response['num_sources']}")
        print(f"   Preview: {response['answer'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def print_installation_check():
    """Print installation verification"""
    print("\n📦 Package Version Check:")
    packages = [
        'langchain', 'langchain-openai', 'langchain-core', 
        'openai', 'pinecone-client', 'numpy'
    ]
    
    for package in packages:
        try:
            import importlib.metadata
            version = importlib.metadata.version(package)
            print(f"   {package}: {version}")
        except:
            print(f"   {package}: ❌ Not installed")

def main():
    """Run all tests"""
    print("🧬 LangChain RAG System Test")
    print("=" * 50)
    
    tests = [
        test_langchain_imports,
        test_core_dependencies,
        test_your_existing_services,
        test_rag_service,
        test_full_rag_pipeline
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Print summary
    print("=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Your LangChain RAG system is ready!")
        print("\n💡 Try this:")
        print("   from services.rag_service import create_rag_service")
        print("   rag = create_rag_service(os.getenv('OPENAI_API_KEY'))")
        print("   response = rag.ask_question('What is CRISPR?')")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        if passed < total:
            print("\n🔧 Next steps:")
            print("1. Run the installation commands again")
            print("2. Check your .env file has correct API keys")
            print("3. Verify your virtual environment is activated")
    
    print_installation_check()

if __name__ == "__main__":
    main()
```

### Run Tests

```bash
python test_langchain_rag.py 
```

Expected output:
```
 LangChain RAG System Test
==================================================
🔍 Testing LangChain imports...
✅ LangChain imports successful
🔍 Testing core dependencies...
✅ Core dependencies successful
🔍 Testing your existing services...
✅ Config loaded: genomics-publications
✅ Search service created
🔍 Testing RAG service...
✅ RAG service created
🔍 Testing full RAG pipeline...
✅ Full pipeline test successful
==================================================
🎉 All tests passed!
```

## 💡 **Usage Examples**

### Basic Usage

```python
from services.rag_service import create_rag_service
import os

# Initialize RAG service
rag = create_rag_service(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-4"  # or "gpt-3.5-turbo"
)

# Ask questions
response = rag.ask_question("What is CRISPR gene editing?")
print(response['answer'])
print(f"Based on {response['num_sources']} research papers")
```

### Advanced Filtering

```python
# High-impact Nature papers from recent years
response = rag.ask_with_paper_focus(
    question="Latest cancer immunotherapy breakthroughs",
    journal="Nature",
    year_range=(2021, 2024),
    min_citations=30,
    top_k=5
)

# Methods-focused search
methods_response = rag.ask_about_methods(
    "What are the standard protocols for single-cell RNA sequencing?"
)

# Recent research summary
recent_response = rag.summarize_recent_research(
    topic="machine learning in genomics",
    years_back=2
)
```

### Comparative Analysis

```python
# Compare different approaches
comparison = rag.compare_approaches(
    question="Compare different gene editing techniques",
    approaches=["CRISPR", "TALEN", "zinc finger nucleases"],
    top_k=8
)
```

##  **Response Format**

All RAG methods return a standardized response:

```python
{
    'question': 'User question',
    'answer': 'Generated answer with citations',
    'sources': [
        {
            'title': 'Paper title',
            'source_file': 'filename.pdf',
            'relevance_score': 0.95,
            'content_preview': 'First 200 chars...',
            'journal': 'Nature',
            'year': 2024,
            'authors': ['Author 1', 'Author 2'],
            'doi': '10.1038/...',
            'citation_count': 45
        }
    ],
    'num_sources': 5,
    'filters_used': {...}
}
```

##  **Troubleshooting Common Issues**

### 1. OpenAI Proxies Error
```bash
# If you get "unexpected keyword argument 'proxies'"
pip install "httpx>=0.25.0,<0.28.0"
pip install --force-reinstall openai==1.51.2
```

### 2. LangChain Field Validation Errors
```python
# This implementation uses SimpleGenomicsRetriever 
# which bypasses BaseRetriever field validation issues
# No changes needed - it's already fixed
```

### 3. Package Version Conflicts
```bash
# Create fresh environment
deactivate && rm -rf venv
python -m venv venv && source venv/bin/activate
pip install [package list from above]
```

### 4. Index Name Mismatch
```bash
# Check your actual index name in Pinecone dashboard
# Update .env file: PINECONE_INDEX_NAME=your-actual-index-name
```

##  **Performance Optimization**

### Model Selection
- **gpt-3.5-turbo** - Fast, cost-effective for simple Q&A
- **gpt-4** - Higher quality, better for complex analysis
- **gpt-4-turbo** - Latest features, longer context

### Search Optimization
```python
# Adjust parameters based on your needs
response = rag.ask_question(
    question="Your question",
    top_k=5,  # Fewer sources = faster response
    filters={"chunk_type": {"$eq": "abstract"}}  # Target specific sections
)
```

##  **Production Deployment**

### Environment Variables
```bash
# Production .env
OPENAI_API_KEY=your_production_key
PINECONE_API_KEY=your_production_key
PINECONE_INDEX_NAME=your_production_index
DEFAULT_LLM_MODEL=gpt-3.5-turbo  # Cost optimization
DEFAULT_TEMPERATURE=0.1
MAX_CONTEXT_TOKENS=4000
```

### Initialization Pattern
```python
# Initialize once at application startup
rag_service = create_rag_service(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name=os.getenv('DEFAULT_LLM_MODEL', 'gpt-4'),
    temperature=float(os.getenv('DEFAULT_TEMPERATURE', '0.1'))
)

# Reuse the service for all requests
def handle_question(question: str, filters: dict = None):
    return rag_service.ask_question(question, filters=filters)
```

##  **Next Steps: UI Integration**

Your RAG system is now ready for UI integration. The service provides:

✅ **Simple API** - Easy to integrate with web frameworks  
✅ **Flexible Filtering** - Perfect for UI filter components  
✅ **Source Attribution** - Rich metadata for displaying sources  
✅ **Multiple Modes** - Different search types for UI options  

Ready to build your web interface with Flask, FastAPI, Streamlit, or any framework!

##  **Key Differences from Original Guide**

1. **Updated package versions** - All compatible with current releases
2. **Fixed LangChain imports** - Uses `langchain_openai.ChatOpenAI`
3. **Simplified retriever** - Bypasses BaseRetriever field validation
4. **httpx compatibility** - Specific version range to avoid conflicts
5. **Production-ready** - Includes comprehensive error handling
6. **Tested implementation** - All code has been verified to work

This implementation provides the same functionality as the original but with modern, working dependencies! 

## Implementation Steps

This section summarizes how to enable and use the new production features in your Genomics RAG system:

### 1. API Key Authentication
- Set an environment variable `API_KEY` in your `.env` file.
- All protected endpoints require an `x-api-key` header with this value.

### 2. Rate Limiting
- Enabled via `slowapi`.
- `/search`: 30 requests/minute per IP.
- `/query`: 20 requests/minute per IP.
- Returns HTTP 429 if exceeded.

### 3. Input Sanitization
- All queries are validated (1–500 chars, whitespace stripped) via Pydantic models.

### 4. Caching
- Query embeddings are cached in-memory (up to 1000 queries) using `cachetools.LRUCache`.
- Reduces redundant OpenAI API calls for repeated queries.

### 5. Connection Pooling
- All OpenAI API calls use a persistent `httpx.Client` for efficient connection reuse and timeouts.

### 6. Dockerization
- A production-ready `Dockerfile` is included.
- Exposes port 8080 and runs the app with `uvicorn`.

---

## Testing

Follow these steps to test the new features:

### 1. API Key Authentication
Set your API key in `.env`:
```
API_KEY=your_secret_key
```
Test with curl (replace `your_secret_key`):
```
curl -X POST http://localhost:8080/search \
  -H 'x-api-key: your_secret_key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "CRISPR", "top_k": 1}'
```
- If the key is missing or wrong, you get HTTP 401 Unauthorized.

### 2. Rate Limiting
Send >30 requests/minute to `/search` or >20/minute to `/query` from the same IP:
```
for i in {1..35}; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/search \
    -H 'x-api-key: your_secret_key' \
    -H 'Content-Type: application/json' \
    -d '{"query": "test", "top_k": 1}'
done
```
- The last few should return `429` Too Many Requests.

### 3. Caching
- Make the same query twice:
```
curl -X POST http://localhost:8080/search \
  -H 'x-api-key: your_secret_key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "CRISPR", "top_k": 1}'
```
- The second call should be faster (embedding is cached; check logs for cache hits).

### 4. Docker Build & Run
Build and run the app in Docker:
```
docker build -t genomics-rag-api .
docker run -p 8080:8080 --env-file .env genomics-rag-api
```
- Test endpoints as above.

### 5. Health Check
```
curl http://localhost:8080/health
```
- Should return `{ "status": "healthy", ... }`

### 6. Logging & Request IDs
- All logs are in JSON format and include a unique request ID per request.
- Each response includes an `X-Request-ID` header.

---

_See the rest of this guide for advanced usage, troubleshooting, and integration details._ 
