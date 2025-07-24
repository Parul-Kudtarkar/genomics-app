from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
import logging
from enum import Enum
from dotenv import load_dotenv
import json

load_dotenv()

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotproduct"

class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class PerformanceConfig:
    batch_size: int = 100
    max_concurrent_requests: int = 10
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    pool_size: int = 10

@dataclass
class MonitoringConfig:
    enable_metrics: bool = True
    log_queries: bool = True
    log_performance: bool = True
    metrics_retention_days: int = 30

@dataclass
class CircuitBreakerConfig:
    enabled: bool = False
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    expected_exception: str = "PineconeException"

@dataclass
class DeadLetterQueueConfig:
    enabled: bool = False
    max_queue_size: int = 1000

@dataclass
class AlertThresholds:
    index_fullness: float = 0.85
    query_latency_p95: int = 1000  # ms
    error_rate: float = 0.05

@dataclass
class HealthCheckConfig:
    interval: int = 30  # seconds

@dataclass
class PineconeConfig:
    # Core configuration
    api_key: str
    index_name: str
    dimension: int = 1536
    metric: MetricType = MetricType.COSINE
    
    # Serverless configuration
    cloud: CloudProvider = CloudProvider.AWS
    region: str = "us-east-1"
    
    # Legacy pod configuration (deprecated for free tier)
    environment: Optional[str] = None
    pod_type: Optional[str] = None
    replicas: int = 1
    shards: int = 1
    
    # Enhanced configuration
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Reliability and advanced monitoring
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    dead_letter_queue_config: DeadLetterQueueConfig = field(default_factory=DeadLetterQueueConfig)
    health_check_config: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    
    # Security and validation
    validate_vectors: bool = True
    max_vector_dimension: int = 2048
    min_vector_dimension: int = 1
    
    # Metadata configuration
    allowed_metadata_keys: Optional[List[str]] = None
    metadata_size_limit: int = 10240  # 10KB per metadata
    
    @classmethod
    def from_env(cls) -> 'PineconeConfig':
        """Load configuration from environment variables with enhanced validation"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Validate API key format
        if not api_key.startswith(('sk-', 'pk-')):
            logger.warning("API key format may be incorrect. Expected format: sk-... or pk-...")
        
        # Parse metric type
        metric_str = os.getenv("PINECONE_METRIC", "cosine").lower()
        try:
            metric = MetricType(metric_str)
        except ValueError:
            logger.warning(f"Invalid metric type: {metric_str}. Using cosine.")
            metric = MetricType.COSINE
        
        # Parse cloud provider
        cloud_str = os.getenv("PINECONE_CLOUD", "aws").lower()
        try:
            cloud = CloudProvider(cloud_str)
        except ValueError:
            logger.warning(f"Invalid cloud provider: {cloud_str}. Using AWS.")
            cloud = CloudProvider.AWS
        
        # Enhanced retry configuration
        retry_config = RetryConfig(
            max_retries=int(os.getenv("PINECONE_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("PINECONE_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("PINECONE_MAX_DELAY", "60.0")),
            exponential_base=float(os.getenv("PINECONE_EXPONENTIAL_BASE", "2.0")),
            jitter=os.getenv("PINECONE_JITTER", "true").lower() == "true"
        )
        
        # Enhanced performance configuration
        performance_config = PerformanceConfig(
            batch_size=int(os.getenv("PINECONE_BATCH_SIZE", "100")),
            max_concurrent_requests=int(os.getenv("PINECONE_MAX_CONCURRENT", "10")),
            connection_timeout=float(os.getenv("PINECONE_CONNECTION_TIMEOUT", "30.0")),
            read_timeout=float(os.getenv("PINECONE_READ_TIMEOUT", "60.0")),
            pool_size=int(os.getenv("PINECONE_POOL_SIZE", "10"))
        )
        
        # Enhanced monitoring configuration
        monitoring_config = MonitoringConfig(
            enable_metrics=os.getenv("PINECONE_ENABLE_METRICS", "true").lower() == "true",
            log_queries=os.getenv("PINECONE_LOG_QUERIES", "true").lower() == "true",
            log_performance=os.getenv("PINECONE_LOG_PERFORMANCE", "true").lower() == "true",
            metrics_retention_days=int(os.getenv("PINECONE_METRICS_RETENTION", "30"))
        )
        
        # Circuit breaker config
        circuit_breaker_config = CircuitBreakerConfig(
            enabled=os.getenv("PINECONE_CIRCUIT_BREAKER_ENABLED", "false").lower() == "true",
            failure_threshold=int(os.getenv("PINECONE_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
            recovery_timeout=int(os.getenv("PINECONE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30")),
            expected_exception=os.getenv("PINECONE_CIRCUIT_BREAKER_EXCEPTION", "PineconeException")
        )
        # Dead letter queue config
        dead_letter_queue_config = DeadLetterQueueConfig(
            enabled=os.getenv("PINECONE_DEAD_LETTER_QUEUE_ENABLED", "false").lower() == "true",
            max_queue_size=int(os.getenv("PINECONE_DEAD_LETTER_QUEUE_MAX_SIZE", "1000"))
        )
        # Health check config
        health_check_config = HealthCheckConfig(
            interval=int(os.getenv("PINECONE_HEALTH_CHECK_INTERVAL", "30"))
        )
        # Alert thresholds
        alert_thresholds_env = os.getenv("PINECONE_ALERT_THRESHOLDS")
        if alert_thresholds_env:
            try:
                alert_thresholds_dict = json.loads(alert_thresholds_env)
                alert_thresholds = AlertThresholds(
                    index_fullness=alert_thresholds_dict.get("index_fullness", 0.85),
                    query_latency_p95=alert_thresholds_dict.get("query_latency_p95", 1000),
                    error_rate=alert_thresholds_dict.get("error_rate", 0.05)
                )
            except Exception:
                alert_thresholds = AlertThresholds()
        else:
            alert_thresholds = AlertThresholds()
        
        # Parse allowed metadata keys
        allowed_keys_str = os.getenv("PINECONE_ALLOWED_METADATA_KEYS")
        allowed_metadata_keys = None
        if allowed_keys_str:
            try:
                allowed_metadata_keys = json.loads(allowed_keys_str)
            except json.JSONDecodeError:
                logger.warning("Invalid PINECONE_ALLOWED_METADATA_KEYS format. Expected JSON array.")
        
        config = cls(
            api_key=api_key,
            index_name=os.getenv("PINECONE_INDEX_NAME", "genomics-publications"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            metric=metric,
            cloud=cloud,
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            retry_config=retry_config,
            performance_config=performance_config,
            monitoring_config=monitoring_config,
            circuit_breaker_config=circuit_breaker_config,
            dead_letter_queue_config=dead_letter_queue_config,
            health_check_config=health_check_config,
            alert_thresholds=alert_thresholds,
            validate_vectors=os.getenv("PINECONE_VALIDATE_VECTORS", "true").lower() == "true",
            max_vector_dimension=int(os.getenv("PINECONE_MAX_VECTOR_DIMENSION", "2048")),
            min_vector_dimension=int(os.getenv("PINECONE_MIN_VECTOR_DIMENSION", "1")),
            allowed_metadata_keys=allowed_metadata_keys,
            metadata_size_limit=int(os.getenv("PINECONE_METADATA_SIZE_LIMIT", "10240"))
        )
        
        # Validate configuration
        config._validate()
        
        return config
    
    def _validate(self):
        """Validate configuration parameters"""
        if not self.index_name or len(self.index_name.strip()) == 0:
            raise ValueError("Index name cannot be empty")
        
        if not self.index_name.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Index name must contain only alphanumeric characters, hyphens, and underscores")
        
        if self.dimension < self.min_vector_dimension or self.dimension > self.max_vector_dimension:
            raise ValueError(f"Dimension must be between {self.min_vector_dimension} and {self.max_vector_dimension}")
        
        if self.performance_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.retry_config.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
    
    def is_serverless(self) -> bool:
        """Check if this is a serverless configuration"""
        return self.environment is None and self.cloud is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging"""
        return {
            "index_name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "cloud": self.cloud.value,
            "region": self.region,
            "is_serverless": self.is_serverless(),
            "batch_size": self.performance_config.batch_size,
            "max_retries": self.retry_config.max_retries
        }
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported regions for the configured cloud provider"""
        region_map = {
            CloudProvider.AWS: [
                "us-east-1", "us-west-1", "us-west-2", "eu-west-1", 
                "ap-southeast-1", "ap-northeast-1"
            ],
            CloudProvider.GCP: [
                "us-central1", "us-east1", "europe-west1", "asia-northeast1"
            ],
            CloudProvider.AZURE: [
                "eastus", "westus", "northeurope", "southeastasia"
            ]
        }
        return region_map.get(self.cloud, [])
    
    def validate_region(self) -> bool:
        """Validate if the configured region is supported"""
        return self.region in self.get_supported_regions()
