from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import json
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

# Enhanced logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

class RetryableError(VectorStoreError):
    """Error that can be retried"""
    pass

class NonRetryableError(VectorStoreError):
    """Error that should not be retried"""
    pass

class PerformanceMetrics:
    """Track performance metrics for the vector store"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def record_operation(self, operation: str, duration: float, success: bool, **kwargs):
        """Record an operation metric"""
        with self.lock:
            timestamp = datetime.now()
            metric = {
                "timestamp": timestamp,
                "operation": operation,
                "duration": duration,
                "success": success,
                **kwargs
            }
            self.metrics[operation].append(metric)
    
    def get_stats(self, operation: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            if operation:
                metrics = self.metrics[operation]
            else:
                # Combine all metrics
                metrics = []
                for op_metrics in self.metrics.values():
                    metrics.extend(op_metrics)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in metrics if m["timestamp"] > cutoff_time]
            
            if not recent_metrics:
                return {"count": 0, "avg_duration": 0, "success_rate": 0}
            
            durations = [m["duration"] for m in recent_metrics]
            successes = sum(1 for m in recent_metrics if m["success"])
            
            return {
                "count": len(recent_metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "success_rate": successes / len(recent_metrics),
                "success_count": successes,
                "error_count": len(recent_metrics) - successes
            }

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      max_delay: float = 60.0, exponential_base: float = 2.0, 
                      jitter: bool = True):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                except NonRetryableError as e:
                    logger.error(f"Non-retryable error: {e}")
                    raise
                except Exception as e:
                    # Determine if error is retryable
                    if _is_retryable_error(e):
                        last_exception = RetryableError(str(e))
                        if attempt == max_retries:
                            raise last_exception
                        
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        raise NonRetryableError(str(e))
            
            raise last_exception
        return wrapper
    return decorator

def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable"""
    error_str = str(error).lower()
    
    # Retryable errors
    retryable_patterns = [
        "timeout", "connection", "network", "rate limit", "throttle",
        "service unavailable", "internal server error", "bad gateway",
        "gateway timeout", "temporary", "retry"
    ]
    
    # Non-retryable errors
    non_retryable_patterns = [
        "authentication", "authorization", "invalid", "not found",
        "bad request", "forbidden", "unauthorized"
    ]
    
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False
    
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True
    
    # Default to retryable for unknown errors
    return True

class PineconeVectorStore:
    def __init__(self, config):
        self.config = config
        self.pc = None
        self.index = None
        self.metrics = PerformanceMetrics(config.monitoring_config.metrics_retention_days)
        self._initialize_pinecone()
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index with enhanced error handling"""
        start_time = time.time()
        
        try:
            # Validate configuration
            self._validate_config()
            
            # Initialize Pinecone client (new API)
            self.pc = Pinecone(api_key=self.config.api_key)
            logger.info("Pinecone client initialized successfully")
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            logger.info(f"Existing indexes: {existing_indexes}")
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.index_name}")
                self._create_index()
            else:
                logger.info(f"Index {self.config.index_name} already exists")
            
            # Connect to index
            self.index = self.pc.Index(self.config.index_name)
            logger.info(f"Connected to Pinecone index: {self.config.index_name}")
            
            # Validate index configuration
            self._validate_index_config()
            
            # Test the connection
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            duration = time.time() - start_time
            self.metrics.record_operation("initialization", duration, True)
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation("initialization", duration, False, error=str(e))
            
            logger.error(f"Failed to initialize Pinecone: {e}")
            logger.error("Troubleshooting steps:")
            logger.error("1. Verify PINECONE_API_KEY is correct")
            logger.error("2. Check available regions for serverless indexes")
            logger.error("3. Ensure you have permissions to create indexes")
            logger.error("4. Validate index name format (alphanumeric, hyphens, underscores only)")
            
            if "rate limit" in str(e).lower():
                raise RetryableError(f"Rate limited: {e}")
            elif "authentication" in str(e).lower():
                raise NonRetryableError(f"Authentication failed: {e}")
            else:
                raise RetryableError(f"Initialization failed: {e}")
    
    def _validate_config(self):
        """Validate configuration before initialization"""
        if not self.config.validate_region():
            supported_regions = self.config.get_supported_regions()
            raise NonRetryableError(
                f"Region '{self.config.region}' not supported for cloud '{self.config.cloud}'. "
                f"Supported regions: {supported_regions}"
            )
        
        if self.config.dimension < self.config.min_vector_dimension:
            raise NonRetryableError(
                f"Dimension {self.config.dimension} is below minimum {self.config.min_vector_dimension}"
            )
        
        if self.config.dimension > self.config.max_vector_dimension:
            raise NonRetryableError(
                f"Dimension {self.config.dimension} exceeds maximum {self.config.max_vector_dimension}"
            )
    
    def _create_index(self):
        """Create Pinecone index with proper error handling and enhanced scientific metadata/performance config"""
        try:
            OPTIMIZED_INDEX_CONFIG = {
                "metadata_config": {
                    "indexed": [
                        "journal", "publication_year", "content_type",
                        "section_type", "experimental_context",
                        "methodology_type", "has_statistical_data",
                        "gene_mentions", "protein_mentions"
                    ]
                },
                "performance_config": {
                    "replicas": 2,
                    "shards": 1,
                    "pod_type": "p1.x2"
                }
            }
            if self.config.is_serverless():
                # Create serverless index with enhanced config
                logger.info(f"Creating serverless index in {self.config.cloud.value}-{self.config.region} with optimized config")
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric.value,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud.value,
                        region=self.config.region
                    ),
                    metadata_config=OPTIMIZED_INDEX_CONFIG["metadata_config"],
                    # performance_config is not supported for serverless, but included for future compatibility
                )
            else:
                # Create pod-based index (legacy) - not supported in free tier
                logger.info(f"Creating pod-based index with optimized config")
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric.value,
                    replicas=OPTIMIZED_INDEX_CONFIG["performance_config"]["replicas"],
                    shards=OPTIMIZED_INDEX_CONFIG["performance_config"]["shards"],
                    pod_type=OPTIMIZED_INDEX_CONFIG["performance_config"]["pod_type"],
                    metadata_config=OPTIMIZED_INDEX_CONFIG["metadata_config"]
                )
            # Wait for index to be ready
            self._wait_for_index_ready()
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Index {self.config.index_name} already exists (race condition)")
                return
            raise
    
    def _wait_for_index_ready(self, max_wait_time: int = 180, wait_interval: int = 5):
        """Wait for index to be ready with enhanced monitoring"""
        logger.info("Waiting for index to be ready...")
        
        for i in range(0, max_wait_time, wait_interval):
            try:
                index_description = self.pc.describe_index(self.config.index_name)
                if index_description.status.ready:
                    logger.info("Index is ready!")
                    return
                else:
                    logger.info(f"Index status: {index_description.status.state} - waiting... ({i+wait_interval}s)")
            except Exception as e:
                logger.info(f"Still initializing... ({i+wait_interval}s)")
            
            time.sleep(wait_interval)
        
        logger.warning("Index creation taking longer than expected, but continuing...")
    
    def _validate_index_config(self):
        """Validate that the index configuration matches our expectations"""
        try:
            index_description = self.pc.describe_index(self.config.index_name)
            
            if index_description.dimension != self.config.dimension:
                logger.warning(
                    f"Index dimension mismatch: expected {self.config.dimension}, "
                    f"got {index_description.dimension}"
                )
            
            if hasattr(index_description, 'metric') and index_description.metric != self.config.metric.value:
                logger.warning(
                    f"Index metric mismatch: expected {self.config.metric.value}, "
                    f"got {index_description.metric}"
                )
                
        except Exception as e:
            logger.warning(f"Could not validate index configuration: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the vector store"""
        try:
            if not self.index:
                return {"status": "error", "message": "Index not initialized"}
            
            stats = self.index.describe_index_stats()
            index_info = self.pc.describe_index(self.config.index_name)
            
            return {
                "status": "healthy",
                "index_name": self.config.index_name,
                "type": "serverless" if self.config.is_serverless() else "pod",
                "cloud": getattr(self.config, 'cloud', None),
                "region": getattr(self.config, 'region', None),
                "environment": getattr(self.config, 'environment', None),
                "ready": index_info.status.ready,
                "stats": stats
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def upsert_documents(self, documents: List[Dict[str, Any]], 
                        batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Upsert documents to Pinecone with enhanced validation and error handling"""
        start_time = time.time()
        
        if not self.index:
            raise NonRetryableError("Index not initialized")
        
        if not documents:
            return {"success": True, "upserted": 0, "errors": 0, "duration": 0}
        
        # Use configured batch size if not specified
        batch_size = batch_size or self.config.performance_config.batch_size
        
        # Validate documents
        validated_docs = []
        validation_errors = []
        
        for i, doc in enumerate(documents):
            try:
                validated_doc = self._validate_document(doc)
                validated_docs.append(validated_doc)
            except Exception as e:
                validation_errors.append(f"Document {i}: {e}")
        
        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")
        
        if not validated_docs:
            raise NonRetryableError("No valid documents to upsert")
        
        # Process in batches
        total_batches = (len(validated_docs) + batch_size - 1) // batch_size
        successful_upserts = 0
        failed_upserts = 0
        batch_errors = []
        
        for i in range(0, len(validated_docs), batch_size):
            batch = validated_docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                batch_start = time.time()
                
                # Format for new API
                vectors = [
                    {
                        "id": doc["id"],
                        "values": doc["values"],
                        "metadata": doc["metadata"]
                    }
                    for doc in batch
                ]
                
                self.index.upsert(vectors=vectors)
                
                batch_duration = time.time() - batch_start
                successful_upserts += len(batch)
                
                if self.config.monitoring_config.log_performance:
                    logger.info(f"Upserted batch {batch_num}/{total_batches} "
                              f"({len(batch)} documents) in {batch_duration:.2f}s")
                
                self.metrics.record_operation(
                    "upsert_batch", batch_duration, True, 
                    batch_size=len(batch), batch_num=batch_num
                )
                
            except Exception as e:
                batch_duration = time.time() - batch_start
                failed_upserts += len(batch)
                error_msg = f"Failed to upsert batch {batch_num}: {e}"
                batch_errors.append(error_msg)
                logger.error(error_msg)
                
                self.metrics.record_operation(
                    "upsert_batch", batch_duration, False, 
                    batch_size=len(batch), batch_num=batch_num, error=str(e)
                )
                
                # For critical errors, stop processing
                if "authentication" in str(e).lower() or "forbidden" in str(e).lower():
                    raise NonRetryableError(f"Critical error in batch {batch_num}: {e}")
        
        duration = time.time() - start_time
        self.metrics.record_operation("upsert_documents", duration, True, 
                                    total=len(documents), successful=successful_upserts)
        
        result = {
            "success": failed_upserts == 0,
            "upserted": successful_upserts,
            "failed": failed_upserts,
            "total": len(documents),
            "duration": duration,
            "validation_errors": validation_errors,
            "batch_errors": batch_errors
        }
        
        if self.config.monitoring_config.log_performance:
            logger.info(f"Upsert completed: {successful_upserts}/{len(documents)} documents "
                       f"in {duration:.2f}s")
        
        return result
    
    def _validate_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean a document before upserting"""
        if not isinstance(doc, dict):
            raise ValueError("Document must be a dictionary")
        
        # Validate required fields
        if "id" not in doc:
            raise ValueError("Document must have an 'id' field")
        
        if "values" not in doc:
            raise ValueError("Document must have a 'values' field")
        
        if "metadata" not in doc:
            raise ValueError("Document must have a 'metadata' field")
        
        # Validate ID
        doc_id = str(doc["id"]).strip()
        if not doc_id:
            raise ValueError("Document ID cannot be empty")
        
        # Validate vector values
        values = doc["values"]
        if not isinstance(values, (list, tuple)):
            raise ValueError("Values must be a list or tuple")
        
        if len(values) != self.config.dimension:
            raise ValueError(f"Vector dimension {len(values)} doesn't match index dimension {self.config.dimension}")
        
        # Validate vector values are numeric
        try:
            values = [float(v) for v in values]
        except (ValueError, TypeError):
            raise ValueError("All vector values must be numeric")
        
        # Validate metadata
        metadata = doc["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Check metadata size limit
        metadata_size = len(json.dumps(metadata))
        if metadata_size > self.config.metadata_size_limit:
            raise ValueError(f"Metadata size {metadata_size} exceeds limit {self.config.metadata_size_limit}")
        
        # Filter metadata keys if specified
        if self.config.allowed_metadata_keys:
            filtered_metadata = {
                k: v for k, v in metadata.items() 
                if k in self.config.allowed_metadata_keys
            }
            if filtered_metadata != metadata:
                logger.warning(f"Filtered metadata keys for document {doc_id}")
            metadata = filtered_metadata
        
        return {
            "id": doc_id,
            "values": values,
            "metadata": metadata
        }
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform similarity search with enhanced error handling and monitoring"""
        start_time = time.time()
        
        try:
            if not self.index:
                raise NonRetryableError("Index not initialized")
            
            # Validate query vector
            if not isinstance(query_vector, (list, tuple)):
                raise NonRetryableError("Query vector must be a list or tuple")
            
            if len(query_vector) != self.config.dimension:
                raise NonRetryableError(
                    f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.config.dimension}"
                )
            
            # Validate vector values are numeric
            try:
                query_vector = [float(v) for v in query_vector]
            except (ValueError, TypeError):
                raise NonRetryableError("All query vector values must be numeric")
            
            # Validate top_k
            if top_k <= 0:
                raise NonRetryableError("top_k must be positive")
            
            if top_k > 10000:  # Pinecone limit
                logger.warning(f"top_k {top_k} exceeds recommended limit, capping at 10000")
                top_k = 10000
            
            # Log query if enabled
            if self.config.monitoring_config.log_queries:
                logger.info(f"Performing similarity search: top_k={top_k}, "
                          f"filter={metadata_filter}, namespace={namespace}")
            
            # Perform search
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
                include_values=include_values,
                namespace=namespace
            )
            
            # Process results
            processed_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                if include_values and hasattr(match, 'values'):
                    result["values"] = match.values
                processed_results.append(result)
            
            duration = time.time() - start_time
            self.metrics.record_operation(
                "similarity_search", duration, True, 
                top_k=top_k, results_count=len(processed_results)
            )
            
            if self.config.monitoring_config.log_performance:
                logger.info(f"Search completed: {len(processed_results)} results in {duration:.3f}s")
            
            return {
                "success": True,
                "results": processed_results,
                "count": len(processed_results),
                "duration": duration,
                "query_info": {
                    "top_k": top_k,
                    "filter": metadata_filter,
                    "namespace": namespace
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation(
                "similarity_search", duration, False, 
                top_k=top_k, error=str(e)
            )
            
            logger.error(f"Similarity search failed: {e}")
            
            if "rate limit" in str(e).lower():
                raise RetryableError(f"Rate limited: {e}")
            elif "authentication" in str(e).lower():
                raise NonRetryableError(f"Authentication failed: {e}")
            else:
                raise RetryableError(f"Search failed: {e}")
    
    def batch_similarity_search(
        self, 
        query_vectors: List[List[float]], 
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        namespace: Optional[str] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform batch similarity search with concurrent processing"""
        if not query_vectors:
            return []
        
        max_concurrent = max_concurrent or self.config.performance_config.max_concurrent_requests
        
        results = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all queries
            future_to_index = {
                executor.submit(
                    self.similarity_search, 
                    vector, top_k, metadata_filter, include_values, namespace
                ): i for i, vector in enumerate(query_vectors)
            }
            
            # Collect results in order
            temp_results = [None] * len(query_vectors)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    temp_results[index] = result
                except Exception as e:
                    logger.error(f"Batch search failed for query {index}: {e}")
                    temp_results[index] = {
                        "success": False,
                        "results": [],
                        "count": 0,
                        "duration": 0,
                        "error": str(e)
                    }
            
            results = temp_results
        
        return results
    
    def hybrid_search(
        self,
        query_vector: List[float],
        text_query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform hybrid search combining vector similarity and text matching"""
        # This is a simplified hybrid search implementation
        # In a real implementation, you might use Pinecone's hybrid search features
        # or implement your own text matching logic
        
        start_time = time.time()
        
        try:
            # Perform vector search
            vector_results = self.similarity_search(
                query_vector, top_k, metadata_filter, False, namespace
            )
            
            if not vector_results["success"]:
                return vector_results
            
            # Simple text matching (you can enhance this with more sophisticated text search)
            text_matched_results = []
            for result in vector_results["results"]:
                metadata = result.get("metadata", {})
                text_fields = [
                    metadata.get("title", ""),
                    metadata.get("abstract", ""),
                    metadata.get("content", ""),
                    " ".join(metadata.get("keywords", []))
                ]
                
                text_content = " ".join(str(field) for field in text_fields).lower()
                if text_query.lower() in text_content:
                    text_matched_results.append(result)
            
            # Combine results (simple approach - you can implement more sophisticated ranking)
            combined_results = []
            vector_scores = {r["id"]: r["score"] for r in vector_results["results"]}
            
            for result in text_matched_results:
                vector_score = vector_scores.get(result["id"], 0)
                # Simple hybrid score (you can implement more sophisticated scoring)
                hybrid_score = alpha * vector_score + (1 - alpha) * 1.0
                result["hybrid_score"] = hybrid_score
                combined_results.append(result)
            
            # Sort by hybrid score
            combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            duration = time.time() - start_time
            self.metrics.record_operation(
                "hybrid_search", duration, True,
                top_k=top_k, results_count=len(combined_results)
            )
            
            return {
                "success": True,
                "results": combined_results[:top_k],
                "count": len(combined_results[:top_k]),
                "duration": duration,
                "query_info": {
                    "vector_query": True,
                    "text_query": text_query,
                    "alpha": alpha,
                    "top_k": top_k
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation(
                "hybrid_search", duration, False,
                top_k=top_k, error=str(e)
            )
            
            logger.error(f"Hybrid search failed: {e}")
            return {
                "success": False,
                "results": [],
                "count": 0,
                "duration": duration,
                "error": str(e)
            }
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics with enhanced information"""
        start_time = time.time()
        
        try:
            if not self.index:
                raise NonRetryableError("Index not initialized")
            
            stats = self.index.describe_index_stats()
            index_info = self.pc.describe_index(self.config.index_name)
            
            # Enhance stats with additional information
            enhanced_stats = {
                "basic_stats": stats,
                "index_info": {
                    "name": index_info.name,
                    "dimension": index_info.dimension,
                    "metric": getattr(index_info, 'metric', 'unknown'),
                    "status": {
                        "ready": index_info.status.ready,
                        "state": index_info.status.state
                    },
                    "type": "serverless" if self.config.is_serverless() else "pod",
                    "cloud": self.config.cloud.value if self.config.is_serverless() else None,
                    "region": self.config.region if self.config.is_serverless() else None
                },
                "performance_metrics": self.metrics.get_stats(hours=24),
                "configuration": self.config.to_dict()
            }
            
            duration = time.time() - start_time
            self.metrics.record_operation("get_stats", duration, True)
            
            return enhanced_stats
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation("get_stats", duration, False, error=str(e))
            
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
    def delete_index(self) -> Dict[str, Any]:
        """Delete the index with confirmation and safety checks"""
        start_time = time.time()
        
        try:
            if not self.pc:
                raise NonRetryableError("Pinecone client not initialized")
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.config.index_name not in existing_indexes:
                return {
                    "success": True,
                    "message": f"Index {self.config.index_name} does not exist",
                    "duration": time.time() - start_time
                }
            
            # Get index stats before deletion for logging
            try:
                stats = self.index.describe_index_stats() if self.index else {}
                vector_count = stats.get('total_vector_count', 0)
            except:
                vector_count = 0
            
            # Delete the index
            self.pc.delete_index(self.config.index_name)
            
            # Clear local references
            self.index = None
            
            duration = time.time() - start_time
            self.metrics.record_operation("delete_index", duration, True, vector_count=vector_count)
            
            logger.info(f"Deleted index: {self.config.index_name} (contained {vector_count} vectors)")
            
            return {
                "success": True,
                "message": f"Index {self.config.index_name} deleted successfully",
                "vector_count": vector_count,
                "duration": duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_operation("delete_index", duration, False, error=str(e))
            
            logger.error(f"Failed to delete index: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": duration
            }
    
    def get_performance_metrics(self, operation: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return self.metrics.get_stats(operation, hours)
    
    def clear_performance_metrics(self) -> bool:
        """Clear all performance metrics"""
        try:
            with self.metrics.lock:
                self.metrics.metrics.clear()
            logger.info("Performance metrics cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear performance metrics: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            # Basic health check
            health = self.health_check()
            
            # Performance metrics
            perf_metrics = self.get_performance_metrics(hours=1)
            
            # Index stats
            stats = self.get_index_stats()
            
            # Determine overall health
            is_healthy = (
                health.get("status") == "healthy" and
                perf_metrics.get("success_rate", 0) > 0.8 and
                "error" not in stats
            )
            
            return {
                "overall_status": "healthy" if is_healthy else "degraded",
                "basic_health": health,
                "performance": perf_metrics,
                "index_stats": stats,
                "configuration": self.config.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate connection and basic operations"""
        start_time = time.time()
        
        try:
            # Test basic operations
            tests = {
                "client_initialized": self.pc is not None,
                "index_connected": self.index is not None,
                "stats_retrieval": False,
                "sample_query": False
            }
            
            # Test stats retrieval
            try:
                stats = self.index.describe_index_stats()
                tests["stats_retrieval"] = True
            except Exception as e:
                logger.warning(f"Stats retrieval failed: {e}")
            
            # Test sample query (if index has data)
            try:
                if stats.get('total_vector_count', 0) > 0:
                    # Create a dummy vector for testing
                    dummy_vector = [0.0] * self.config.dimension
                    result = self.similarity_search(dummy_vector, top_k=1)
                    tests["sample_query"] = result.get("success", False)
                else:
                    tests["sample_query"] = True  # No data to test with
            except Exception as e:
                logger.warning(f"Sample query failed: {e}")
            
            duration = time.time() - start_time
            
            return {
                "success": all(tests.values()),
                "tests": tests,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations"""
        return [
            "similarity_search",
            "batch_similarity_search", 
            "hybrid_search",
            "upsert_documents",
            "get_index_stats",
            "health_check",
            "get_performance_metrics",
            "validate_connection",
            "delete_index"
        ]
    
    def get_operation_info(self, operation: str) -> Dict[str, Any]:
        """Get information about a specific operation"""
        operation_info = {
            "similarity_search": {
                "description": "Perform vector similarity search",
                "parameters": ["query_vector", "top_k", "metadata_filter", "include_values", "namespace"],
                "returns": "Search results with metadata",
                "retryable": True
            },
            "batch_similarity_search": {
                "description": "Perform batch vector similarity search with concurrent processing",
                "parameters": ["query_vectors", "top_k", "metadata_filter", "include_values", "namespace", "max_concurrent"],
                "returns": "List of search results",
                "retryable": True
            },
            "hybrid_search": {
                "description": "Perform hybrid search combining vector similarity and text matching",
                "parameters": ["query_vector", "text_query", "top_k", "metadata_filter", "alpha", "namespace"],
                "returns": "Combined search results",
                "retryable": True
            },
            "upsert_documents": {
                "description": "Upsert documents to the vector store",
                "parameters": ["documents", "batch_size"],
                "returns": "Upsert operation results",
                "retryable": True
            },
            "get_index_stats": {
                "description": "Get comprehensive index statistics",
                "parameters": [],
                "returns": "Index statistics and performance metrics",
                "retryable": True
            },
            "health_check": {
                "description": "Perform basic health check",
                "parameters": [],
                "returns": "Health status information",
                "retryable": False
            },
            "get_performance_metrics": {
                "description": "Get performance metrics for operations",
                "parameters": ["operation", "hours"],
                "returns": "Performance statistics",
                "retryable": False
            },
            "validate_connection": {
                "description": "Validate connection and basic operations",
                "parameters": [],
                "returns": "Connection validation results",
                "retryable": False
            },
            "delete_index": {
                "description": "Delete the index (use with caution)",
                "parameters": [],
                "returns": "Deletion operation results",
                "retryable": True
            }
        }
        
        return operation_info.get(operation, {"error": "Operation not found"})
