#!/usr/bin/env python3
"""
Enhanced Pinecone Vector Database Setup Script

This script provides comprehensive setup, validation, and testing for the Pinecone vector database.
It includes enhanced error handling, configuration validation, and performance testing.
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.vector_db import PineconeConfig
from services.vector_store import PineconeVectorStore, VectorStoreError

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pinecone_setup.log')
    ]
)
logger = logging.getLogger(__name__)

class SetupManager:
    """Manages the Pinecone setup process with comprehensive validation"""
    
    def __init__(self, config: PineconeConfig):
        self.config = config
        self.vector_store = None
        self.setup_results = {}
    
    def run_full_setup(self) -> Dict[str, Any]:
        """Run complete setup with validation and testing"""
        logger.info("🚀 Starting Enhanced Pinecone Setup")
        logger.info("=" * 60)
        
        try:
            # Step 1: Validate configuration
            self._validate_configuration()
            
            # Step 2: Initialize vector store
            self._initialize_vector_store()
            
            # Step 3: Perform health checks
            self._perform_health_checks()
            
            # Step 4: Run performance tests
            self._run_performance_tests()
            
            # Step 5: Generate setup report
            report = self._generate_setup_report()
            
            logger.info("✅ Setup completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            self._generate_error_report(str(e))
            raise
    
    def _validate_configuration(self):
        """Validate configuration before setup"""
        logger.info("🔍 Validating configuration...")
        
        validation_results = {
            "api_key_valid": bool(self.config.api_key and len(self.config.api_key) > 10),
            "index_name_valid": bool(self.config.index_name and len(self.config.index_name) > 0),
            "dimension_valid": self.config.min_vector_dimension <= self.config.dimension <= self.config.max_vector_dimension,
            "region_supported": self.config.validate_region(),
            "serverless_config": self.config.is_serverless()
        }
        
        # Check for validation failures
        failed_validations = [k for k, v in validation_results.items() if not v]
        
        if failed_validations:
            error_msg = f"Configuration validation failed: {failed_validations}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
        
        logger.info("✅ Configuration validation passed")
        self.setup_results["validation"] = validation_results
    
    def _initialize_vector_store(self):
        """Initialize the vector store"""
        logger.info("🔧 Initializing vector store...")
        
        try:
            self.vector_store = PineconeVectorStore(self.config)
            logger.info("✅ Vector store initialized successfully")
            
            # Test basic connectivity
            connection_test = self.vector_store.validate_connection()
            if not connection_test["success"]:
                raise VectorStoreError(f"Connection validation failed: {connection_test}")
            
            self.setup_results["initialization"] = {
                "success": True,
                "connection_test": connection_test
            }
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            raise
    
    def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        logger.info("🏥 Performing health checks...")
        
        try:
            # Basic health check
            health = self.vector_store.health_check()
            
            # Comprehensive health status
            health_status = self.vector_store.get_health_status()
            
            # Index stats
            stats = self.vector_store.get_index_stats()
            
            health_results = {
                "basic_health": health,
                "comprehensive_health": health_status,
                "index_stats": stats,
                "overall_healthy": health_status.get("overall_status") == "healthy"
            }
            
            if health_results["overall_healthy"]:
                logger.info("✅ Health checks passed")
            else:
                logger.warning("⚠️ Health checks show degraded status")
            
            self.setup_results["health_checks"] = health_results
            
        except Exception as e:
            logger.error(f"❌ Health checks failed: {e}")
            raise
    
    def _run_performance_tests(self):
        """Run basic performance tests"""
        logger.info("⚡ Running performance tests...")
        
        try:
            # Test document upsert
            test_docs = self._create_test_documents(5)
            upsert_result = self.vector_store.upsert_documents(test_docs)
            
            # Test similarity search
            query_vector = [0.1] * self.config.dimension
            search_result = self.vector_store.similarity_search(query_vector, top_k=3)
            
            # Get performance metrics
            perf_metrics = self.vector_store.get_performance_metrics(hours=1)
            
            performance_results = {
                "upsert_test": upsert_result,
                "search_test": search_result,
                "performance_metrics": perf_metrics,
                "tests_passed": upsert_result.get("success", False) and search_result.get("success", False)
            }
            
            if performance_results["tests_passed"]:
                logger.info("✅ Performance tests passed")
            else:
                logger.warning("⚠️ Some performance tests failed")
            
            self.setup_results["performance_tests"] = performance_results
            
        except Exception as e:
            logger.error(f"❌ Performance tests failed: {e}")
            # Don't raise here, as this is not critical for setup
    
    def _create_test_documents(self, count: int) -> list:
        """Create test documents for performance testing"""
        import numpy as np
        
        test_docs = []
        for i in range(count):
            doc = {
                "id": f"test_doc_{i}_{int(time.time())}",
                "values": np.random.rand(self.config.dimension).tolist(),
                "metadata": {
                    "title": f"Test Document {i}",
                    "type": "test",
                    "created_at": time.time(),
                    "test_batch": True
                }
            }
            test_docs.append(doc)
        
        return test_docs
    
    def _generate_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup report"""
        logger.info("📊 Generating setup report...")
        
        report = {
            "setup_successful": True,
            "timestamp": time.time(),
            "configuration": self.config.to_dict(),
            "results": self.setup_results,
            "recommendations": self._generate_recommendations(),
            "next_steps": self._get_next_steps()
        }
        
        # Save report to file
        report_file = Path("pinecone_setup_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Setup report saved to: {report_file}")
        return report
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on setup results"""
        recommendations = []
        
        # Check performance metrics
        perf_metrics = self.setup_results.get("performance_tests", {}).get("performance_metrics", {})
        if perf_metrics.get("count", 0) > 0:
            avg_duration = perf_metrics.get("avg_duration", 0)
            if avg_duration > 1.0:
                recommendations.append("Consider optimizing batch sizes for better performance")
        
        # Check health status
        health_status = self.setup_results.get("health_checks", {}).get("comprehensive_health", {})
        if health_status.get("overall_status") != "healthy":
            recommendations.append("Monitor system health and address any issues")
        
        # Check configuration
        if not self.config.monitoring_config.enable_metrics:
            recommendations.append("Enable metrics collection for better monitoring")
        
        if not recommendations:
            recommendations.append("Setup looks good! Monitor performance and adjust as needed.")
        
        return recommendations
    
    def _get_next_steps(self) -> list:
        """Get next steps after setup"""
        return [
            "Start ingesting your genomics publications",
            "Configure your embedding service",
            "Set up monitoring and alerting",
            "Test search functionality with real data",
            "Optimize performance based on usage patterns"
        ]
    
    def _generate_error_report(self, error: str):
        """Generate error report when setup fails"""
        error_report = {
            "setup_successful": False,
            "timestamp": time.time(),
            "error": error,
            "configuration": self.config.to_dict() if self.config else None,
            "troubleshooting_steps": [
                "Verify your Pinecone API key is correct",
                "Check that your index name follows naming conventions",
                "Ensure your region is supported for serverless indexes",
                "Verify you have sufficient permissions",
                "Check network connectivity to Pinecone"
            ]
        }
        
        error_file = Path("pinecone_setup_error.json")
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        logger.error(f"📄 Error report saved to: {error_file}")

def main():
    """Main setup function with command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Pinecone Vector Database Setup")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration without creating index")
    parser.add_argument("--health-check", action="store_true",
                       help="Only perform health checks on existing setup")
    parser.add_argument("--performance-test", action="store_true",
                       help="Only run performance tests")
    parser.add_argument("--config-file", type=str,
                       help="Path to configuration file (optional, JSON format)")
    parser.add_argument("--report-file", type=str, default="pinecone_setup_report.json",
                       help="Path to save the setup report (default: pinecone_setup_report.json)")
    parser.add_argument("--skip-performance-tests", action="store_true",
                       help="Skip performance tests for faster setup")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config_file:
            logger.info(f"Loading configuration from: {args.config_file}")
            import json
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
            # Convert config_data to PineconeConfig (handle enums)
            from config.vector_db import MetricType, CloudProvider
            config_data["metric"] = MetricType(config_data.get("metric", "cosine"))
            config_data["cloud"] = CloudProvider(config_data.get("cloud", "aws"))
            config = PineconeConfig(**config_data)
        else:
            config = PineconeConfig.from_env()
        logger.info(f"Configuration loaded: {config.to_dict()}")
        
        setup_manager = SetupManager(config)
        
        if args.validate_only:
            logger.info("🔍 Running configuration validation only...")
            setup_manager._validate_configuration()
            logger.info("✅ Configuration validation completed")
            return
        
        if args.health_check:
            logger.info("🏥 Running health checks only...")
            setup_manager._initialize_vector_store()
            setup_manager._perform_health_checks()
            return
        
        if args.performance_test:
            logger.info("⚡ Running performance tests only...")
            setup_manager._initialize_vector_store()
            setup_manager._run_performance_tests()
            return
        
        # Run full setup
        logger.info("Running full setup...")
        setup_manager._validate_configuration()
        setup_manager._initialize_vector_store()
        setup_manager._perform_health_checks()
        if not args.skip_performance_tests:
            setup_manager._run_performance_tests()
        report = setup_manager._generate_setup_report()
        
        # Save report to custom file if specified
        import shutil
        if args.report_file != "pinecone_setup_report.json":
            shutil.copyfile("pinecone_setup_report.json", args.report_file)
            logger.info(f"📄 Setup report also saved to: {args.report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 SETUP SUMMARY")
        print("=" * 60)
        print(f"Status: {'✅ SUCCESS' if report['setup_successful'] else '❌ FAILED'}")
        print(f"Index: {config.index_name}")
        print(f"Type: {'Serverless' if config.is_serverless() else 'Pod-based'}")
        print(f"Region: {config.region}")
        print(f"Dimension: {config.dimension}")
        
        if report['setup_successful']:
            health_status = report['results']['health_checks']['comprehensive_health']['overall_status']
            print(f"Health: {health_status.upper()}")
            
            print("\n📋 Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n📄 Detailed report saved to: {args.report_file}")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import time
    main()
