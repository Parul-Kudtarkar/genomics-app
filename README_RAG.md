# 🧬 Enhanced Genomics RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for genomics research, integrated with your existing Pinecone vector store and data ingestion pipelines.

## 🎯 **Overview**

This enhanced RAG system provides:

- **🤖 AI-Powered Q&A** - Ask questions about genomics research papers
- **🔍 Advanced Filtering** - Filter by journal, author, year, citations, source type
- **📝 Specialized Prompts** - Methods, results, comparative analysis
- **💾 Intelligent Caching** - Fast responses for repeated queries
- **📊 Rich Metadata** - Comprehensive source attribution and confidence scoring
- **⚡ Production Ready** - Error handling, monitoring, and performance optimization

## 🚀 **Quick Start**

### 1. **Installation**

```bash
# Install enhanced dependencies
pip install -r requirements.txt

# Verify installation
python scripts/test_enhanced_rag.py --test environment
```

### 2. **Environment Setup**

Create/update your `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# RAG Configuration
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.1
DEFAULT_TOP_K=5
MAX_CONTEXT_TOKENS=4000
ENABLE_CACHING=true
CACHE_SIZE=1000
RAG_TIMEOUT=30
```

### 3. **Basic Usage**

```python
from services.rag_service import create_rag_service

# Create RAG service
rag = create_rag_service()

# Ask a question
response = rag.ask_question("What is CRISPR gene editing?")

print(f"Answer: {response.answer}")
print(f"Sources: {response.num_sources}")
print(f"Confidence: {response.confidence_score:.2f}")
```

## 🔧 **Core Features**

### **1. Basic Question Answering**

```python
# Simple question
response = rag.ask_question("What is gene therapy?", top_k=5)

# With custom filters
response = rag.ask_question(
    "CRISPR applications in medicine",
    filters={"journal": {"$eq": "Nature"}},
    top_k=3
)
```

### **2. Advanced Filtering**

```python
# High-impact Nature papers
response = rag.ask_with_paper_focus(
    question="Latest CRISPR developments",
    journal="Nature",
    min_citations=50,
    top_k=5
)

# Recent research
response = rag.ask_with_paper_focus(
    question="Recent advances in cancer genomics",
    year_range=(2020, 2024),
    top_k=5
)

# Author-specific research
response = rag.ask_with_paper_focus(
    question="Gene editing techniques",
    author="Doudna",
    top_k=5
)
```

### **3. Specialized Prompts**

```python
# Methods-focused questions
response = rag.ask_about_methods(
    "What are the standard protocols for DNA sequencing?",
    top_k=5
)

# Results-focused questions
response = rag.ask_about_results(
    "What are the key findings in cancer genomics?",
    top_k=5
)

# Comparative analysis
response = rag.compare_approaches(
    "Compare different gene editing techniques",
    approaches=["CRISPR", "TALEN", "Zinc Finger Nucleases"],
    top_k=8
)
```

### **4. Recent Research Summaries**

```python
# Summarize recent research
response = rag.summarize_recent_research(
    topic="machine learning in genomics",
    years_back=3,
    max_papers=10
)
```

### **5. Document-Specific Search**

```python
# Search within a specific document
response = rag.search_by_document(
    doc_id="your_document_id",
    question="What are the main findings?",
    top_k=3
)
```

## 📊 **Response Format**

All RAG methods return a structured `RAGResponse` object:

```python
@dataclass
class RAGResponse:
    question: str                    # Original question
    answer: str                     # Generated answer
    sources: List[Dict[str, Any]]   # Source papers with metadata
    num_sources: int                # Number of sources used
    search_query: str               # Search query used
    model_used: str                 # LLM model used
    processing_time: float          # Processing time in seconds
    filters_used: Optional[Dict]    # Filters applied
    confidence_score: Optional[float] # Confidence score (0-1)
    error: Optional[str]            # Error message if any
    metadata: Dict[str, Any]        # Additional metadata
```

### **Source Metadata**

Each source includes comprehensive metadata:

```python
{
    'title': 'Paper title',
    'source_file': 'filename.pdf',
    'relevance_score': 0.95,
    'content_preview': 'First 200 chars...',
    'journal': 'Nature',
    'year': 2024,
    'authors': ['Author 1', 'Author 2'],
    'doi': '10.1038/...',
    'citation_count': 45,
    'keywords': ['CRISPR', 'gene editing'],
    'publication_date': '2024-01-15',
    'source_type': 'pubmed',
    'paper_id': 'unique_id',
    'chunk_index': 0
}
```

## ⚙️ **Configuration**

### **RAGConfig Options**

```python
from services.rag_service import RAGConfig

config = RAGConfig(
    # LLM Configuration
    model_name="gpt-4",              # or "gpt-3.5-turbo"
    temperature=0.1,                 # Creativity (0-1)
    max_tokens=4000,                 # Max response length
    
    # Search Configuration
    default_top_k=5,                 # Default sources to retrieve
    max_context_tokens=4000,         # Max context length
    chunk_overlap=200,               # Chunk overlap for processing
    
    # Performance Configuration
    enable_caching=True,             # Enable response caching
    cache_size=1000,                 # Cache size
    timeout_seconds=30,              # Request timeout
    
    # Prompt Configuration
    include_sources=True,            # Include source information
    include_metadata=True            # Include metadata in responses
)
```

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM_MODEL` | `gpt-4` | LLM model to use |
| `DEFAULT_TEMPERATURE` | `0.1` | Model temperature |
| `DEFAULT_TOP_K` | `5` | Default number of sources |
| `MAX_CONTEXT_TOKENS` | `4000` | Max context length |
| `ENABLE_CACHING` | `true` | Enable response caching |
| `CACHE_SIZE` | `1000` | Cache size |
| `RAG_TIMEOUT` | `30` | Request timeout |

## 🧪 **Testing**

### **Run All Tests**

```bash
python scripts/test_enhanced_rag.py
```

### **Run Specific Tests**

```bash
# Test basic functionality
python scripts/test_enhanced_rag.py --test basic_qa

# Test advanced filtering
python scripts/test_enhanced_rag.py --test advanced_filtering

# Test performance
python scripts/test_enhanced_rag.py --test performance
```

### **Test Categories**

- `environment` - Environment and configuration setup
- `initialization` - RAG service initialization
- `basic_qa` - Basic question answering
- `advanced_filtering` - Advanced filtering capabilities
- `specialized_prompts` - Specialized prompt templates
- `caching` - Caching functionality
- `error_handling` - Error handling capabilities
- `performance` - Performance and metrics collection

## 📖 **Usage Examples**

### **Complete Examples**

Run comprehensive examples:

```bash
# Run all examples
python example_rag_usage.py

# Run specific example
python example_rag_usage.py --example basic_qa
```

### **Example Categories**

- `basic_qa` - Basic question answering
- `advanced_filtering` - Advanced filtering capabilities
- `specialized_prompts` - Specialized prompt templates
- `comparative_analysis` - Comparative analysis
- `recent_research` - Recent research summaries
- `document_search` - Document-specific search
- `custom_config` - Custom configuration
- `performance` - Performance analysis

## 🔍 **Integration with Existing Setup**

### **Vector Store Integration**

The RAG system integrates seamlessly with your existing Pinecone setup:

```python
from config.vector_db import PineconeConfig
from services.vector_store import PineconeVectorStore
from services.rag_service import create_rag_service

# Use existing vector store
config = PineconeConfig.from_env()
vector_store = PineconeVectorStore(config)

# Create RAG service with existing vector store
rag = create_rag_service(vector_store=vector_store)
```

### **Data Ingestion Compatibility**

Works with your existing ingestion pipelines:

- **Text Pipeline** - `text_ingestion_pipeline.py`
- **XML Pipeline** - `xml_ingestion_pipeline.py`
- **PDF Pipeline** - `professional_pdf_converter.py`

The RAG system automatically uses the metadata from your ingested documents.

## 📈 **Performance Optimization**

### **Caching**

Enable caching for faster responses:

```python
config = RAGConfig(enable_caching=True, cache_size=1000)
rag = create_rag_service(config=config)
```

### **Model Selection**

Choose the right model for your needs:

```python
# Fast, cost-effective
fast_config = RAGConfig(model_name="gpt-3.5-turbo", default_top_k=3)

# High-quality, detailed
detailed_config = RAGConfig(model_name="gpt-4", default_top_k=8)
```

### **Context Optimization**

Optimize context length for better performance:

```python
config = RAGConfig(
    max_context_tokens=2000,  # Shorter for speed
    default_top_k=3           # Fewer sources
)
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **"OpenAI API key required"**
   ```bash
   # Check your .env file
   echo $OPENAI_API_KEY
   ```

2. **"Pinecone index not found"**
   ```bash
   # Verify your index name
   python scripts/setup_vector_db.py --health-check
   ```

3. **"No sources found"**
   ```bash
   # Check if you have data in your index
   python scripts/analytics.py
   ```

### **Performance Issues**

1. **Slow responses**
   - Enable caching: `ENABLE_CACHING=true`
   - Use faster model: `DEFAULT_LLM_MODEL=gpt-3.5-turbo`
   - Reduce sources: `DEFAULT_TOP_K=3`

2. **High costs**
   - Use gpt-3.5-turbo instead of gpt-4
   - Reduce max_tokens
   - Enable caching to avoid repeated queries

### **Debug Mode**

Enable verbose logging:

```bash
python scripts/test_enhanced_rag.py --verbose
```

## 📊 **Monitoring and Analytics**

### **Service Statistics**

```python
stats = rag.get_service_statistics()
print(f"Vector store: {stats['vector_store']['total_vectors']} vectors")
print(f"Cache size: {stats['cache']['size']}")
print(f"Model: {stats['config']['model']}")
```

### **Performance Metrics**

The system tracks:
- Processing time per query
- Number of sources retrieved
- Cache hit rates
- Confidence scores
- Error rates

## 🔒 **Security and Best Practices**

### **API Key Security**

- Store API keys in environment variables
- Never commit keys to version control
- Use different keys for development/production

### **Rate Limiting**

- Monitor OpenAI API usage
- Implement request throttling if needed
- Use caching to reduce API calls

### **Data Privacy**

- The system doesn't store user queries
- Cached responses are in-memory only
- Source papers are from your existing vector store

## 🚀 **Production Deployment**

### **Docker Deployment**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### **Environment Variables for Production**

```bash
# Production .env
OPENAI_API_KEY=your_production_key
PINECONE_API_KEY=your_production_key
PINECONE_INDEX_NAME=your_production_index
DEFAULT_LLM_MODEL=gpt-3.5-turbo  # Cost optimization
ENABLE_CACHING=true
CACHE_SIZE=2000
RAG_TIMEOUT=60
```

## 📚 **API Integration**

### **FastAPI Integration**

```python
from fastapi import FastAPI
from services.rag_service import create_rag_service

app = FastAPI()
rag = create_rag_service()

@app.post("/query")
async def query(request: QueryRequest):
    response = rag.ask_question(
        request.question,
        top_k=request.top_k,
        filters=request.filters
    )
    return response
```

### **Flask Integration**

```python
from flask import Flask, request, jsonify
from services.rag_service import create_rag_service

app = Flask(__name__)
rag = create_rag_service()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    response = rag.ask_question(
        data['question'],
        top_k=data.get('top_k', 5)
    )
    return jsonify(response.__dict__)
```

## 🤝 **Contributing**

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Test with existing data ingestion pipelines

## 📄 **License**

This project is part of your genomics research system. Use according to your existing license terms.

---

**Ready to explore genomics research with AI-powered insights! 🧬🤖** 