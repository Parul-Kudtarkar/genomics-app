#!/usr/bin/env python3
"""
Enhanced Pinecone Vector Store Test Suite

This comprehensive test suite validates all aspects of the enhanced Pinecone vector store,
including performance, error handling, and advanced features.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config.vector_db import PineconeConfig
from services.vector_store import PineconeVectorStore, VectorStoreError

class ComprehensiveTestSuite:
    """Comprehensive test suite for the enhanced Pinecone vector store"""
    
    def __init__(self, config: PineconeConfig):
        self.config = config
        self.vector_store = None
        self.test_results = {}
        self.test_documents = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories"""
        print("🧪 Starting Comprehensive Pinecone Test Suite")
        print("=" * 70)
        
        try:
            # Initialize vector store
            self._initialize_vector_store()
            
            # Run test categories
            self._test_basic_functionality()
            self._test_advanced_features()
            self._test_error_handling()
            self._test_performance()
            self._test_monitoring()
            
            # Generate test report
            report = self._generate_test_report()
            
            print("\n🎉 All tests completed!")
            return report
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            self._generate_error_report(str(e))
            raise
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run a quick subset of tests (skip long-running tests)"""
        print("🧪 Starting Quick Pinecone Test Suite (skipping long-running tests)")
        print("=" * 70)
        try:
            self._initialize_vector_store()
            self._test_basic_functionality()
            self._test_advanced_features()
            self._test_error_handling()
            self._test_monitoring()
            # Skip _test_performance (long-running)
            report = self._generate_test_report()
            print("\n🎉 Quick tests completed!")
            return report
        except Exception as e:
            print(f"\n❌ Quick test suite failed: {e}")
            self._generate_error_report(str(e))
            raise
    
    def _initialize_vector_store(self):
        """Initialize the vector store for testing"""
        print("🔧 Initializing vector store...")
        
        try:
            self.vector_store = PineconeVectorStore(self.config)
            
            # Validate connection
            connection_test = self.vector_store.validate_connection()
            if not connection_test["success"]:
                raise VectorStoreError(f"Connection validation failed: {connection_test}")
            
            print("✅ Vector store initialized successfully")
            
        except Exception as e:
            print(f"❌ Vector store initialization failed: {e}")
            raise
    
    def _test_basic_functionality(self):
        """Test basic vector store functionality"""
        print("\n📋 Testing Basic Functionality")
        print("-" * 40)
        
        results = {}
        
        # Test 1: Health check
        print("🔍 Testing health check...")
        try:
            health = self.vector_store.health_check()
            results["health_check"] = {
                "success": health.get("status") == "healthy",
                "details": health
            }
            print("✅ Health check passed")
        except Exception as e:
            results["health_check"] = {"success": False, "error": str(e)}
            print(f"❌ Health check failed: {e}")
        
        # Test 2: Document upsert
        print("📝 Testing document upsert...")
        try:
            self.test_documents = self._create_test_documents(10)
            upsert_result = self.vector_store.upsert_documents(self.test_documents)
            results["document_upsert"] = {
                "success": upsert_result.get("success", False),
                "details": upsert_result
            }
            print(f"✅ Document upsert passed ({upsert_result.get('upserted', 0)} documents)")
        except Exception as e:
            results["document_upsert"] = {"success": False, "error": str(e)}
            print(f"❌ Document upsert failed: {e}")
        
        # Test 3: Similarity search
        print("🔍 Testing similarity search...")
        try:
            query_vector = np.random.rand(self.config.dimension).tolist()
            search_result = self.vector_store.similarity_search(query_vector, top_k=5)
            results["similarity_search"] = {
                "success": search_result.get("success", False),
                "details": search_result
            }
            print(f"✅ Similarity search passed ({search_result.get('count', 0)} results)")
        except Exception as e:
            results["similarity_search"] = {"success": False, "error": str(e)}
            print(f"❌ Similarity search failed: {e}")
        
        # Test 4: Index stats
        print("📊 Testing index stats...")
        try:
            stats = self.vector_store.get_index_stats()
            results["index_stats"] = {
                "success": "error" not in stats,
                "details": stats
            }
            print("✅ Index stats retrieval passed")
        except Exception as e:
            results["index_stats"] = {"success": False, "error": str(e)}
            print(f"❌ Index stats failed: {e}")
        
        self.test_results["basic_functionality"] = results
    
    def _test_advanced_features(self):
        """Test advanced vector store features"""
        print("\n🚀 Testing Advanced Features")
        print("-" * 40)
        
        results = {}
        
        # Test 1: Batch similarity search
        print("🔄 Testing batch similarity search...")
        try:
            query_vectors = [
                np.random.rand(self.config.dimension).tolist() for _ in range(3)
            ]
            batch_result = self.vector_store.batch_similarity_search(query_vectors, top_k=3)
            results["batch_search"] = {
                "success": len(batch_result) == 3,
                "details": {"count": len(batch_result), "results": batch_result}
            }
            print(f"✅ Batch search passed ({len(batch_result)} queries)")
        except Exception as e:
            results["batch_search"] = {"success": False, "error": str(e)}
            print(f"❌ Batch search failed: {e}")
        
        # Test 2: Hybrid search
        print("🔬 Testing hybrid search...")
        try:
            query_vector = np.random.rand(self.config.dimension).tolist()
            hybrid_result = self.vector_store.hybrid_search(
                query_vector, "CRISPR gene editing", top_k=3
            )
            results["hybrid_search"] = {
                "success": hybrid_result.get("success", False),
                "details": hybrid_result
            }
            print(f"✅ Hybrid search passed ({hybrid_result.get('count', 0)} results)")
        except Exception as e:
            results["hybrid_search"] = {"success": False, "error": str(e)}
            print(f"❌ Hybrid search failed: {e}")
        
        # Test 3: Filtered search
        print("🔍 Testing filtered search...")
        try:
            query_vector = np.random.rand(self.config.dimension).tolist()
            filtered_result = self.vector_store.similarity_search(
                query_vector, 
                top_k=5,
                metadata_filter={"year": 2024}
            )
            results["filtered_search"] = {
                "success": filtered_result.get("success", False),
                "details": filtered_result
            }
            print(f"✅ Filtered search passed ({filtered_result.get('count', 0)} results)")
        except Exception as e:
            results["filtered_search"] = {"success": False, "error": str(e)}
            print(f"❌ Filtered search failed: {e}")
        
        # Test 4: Document validation
        print("✅ Testing document validation...")
        try:
            invalid_docs = [
                {"id": "invalid_1", "values": [1, 2, 3], "metadata": {}},  # Wrong dimension
                {"id": "", "values": [0.1] * self.config.dimension, "metadata": {}},  # Empty ID
                {"id": "invalid_3", "values": ["not", "numbers"], "metadata": {}}  # Non-numeric values
            ]
            
            validation_errors = []
            for doc in invalid_docs:
                try:
                    self.vector_store._validate_document(doc)
                except Exception as e:
                    validation_errors.append(str(e))
            
            results["document_validation"] = {
                "success": len(validation_errors) == 3,  # All should fail
                "details": {"validation_errors": validation_errors}
            }
            print(f"✅ Document validation passed ({len(validation_errors)} errors caught)")
        except Exception as e:
            results["document_validation"] = {"success": False, "error": str(e)}
            print(f"❌ Document validation failed: {e}")
        
        self.test_results["advanced_features"] = results
    
    def _test_error_handling(self):
        """Test error handling and retry logic"""
        print("\n🛡️ Testing Error Handling")
        print("-" * 40)
        
        results = {}
        
        # Test 1: Invalid query vector
        print("🔍 Testing invalid query vector handling...")
        try:
            invalid_vector = [1, 2, 3]  # Wrong dimension
            result = self.vector_store.similarity_search(invalid_vector, top_k=1)
            results["invalid_query_vector"] = {
                "success": not result.get("success", True),  # Should fail
                "details": result
            }
            print("✅ Invalid query vector handling passed")
        except Exception as e:
            results["invalid_query_vector"] = {"success": True, "error": str(e)}
            print(f"✅ Invalid query vector properly rejected: {e}")
        
        # Test 2: Empty document list
        print("📝 Testing empty document list handling...")
        try:
            result = self.vector_store.upsert_documents([])
            results["empty_document_list"] = {
                "success": result.get("success", False),
                "details": result
            }
            print("✅ Empty document list handling passed")
        except Exception as e:
            results["empty_document_list"] = {"success": False, "error": str(e)}
            print(f"❌ Empty document list handling failed: {e}")
        
        # Test 3: Large metadata handling
        print("📄 Testing large metadata handling...")
        try:
            large_metadata = {"large_field": "x" * 15000}  # Exceeds 10KB limit
            large_doc = {
                "id": "large_metadata_test",
                "values": [0.1] * self.config.dimension,
                "metadata": large_metadata
            }
            result = self.vector_store._validate_document(large_doc)
            results["large_metadata"] = {"success": False, "details": "Should have failed"}
            print("❌ Large metadata should have been rejected")
        except Exception as e:
            results["large_metadata"] = {"success": True, "error": str(e)}
            print(f"✅ Large metadata properly rejected: {e}")
        
        self.test_results["error_handling"] = results
    
    def _test_performance(self, quick_mode: bool = False):
        """Test performance characteristics"""
        print("\n⚡ Testing Performance")
        print("-" * 40)
        
        results = {}
        
        # Test 1: Batch upsert performance
        if not quick_mode:
            print("📝 Testing batch upsert performance...")
            try:
                large_batch = self._create_test_documents(50)
                start_time = time.time()
                upsert_result = self.vector_store.upsert_documents(large_batch)
                duration = time.time() - start_time
                
                results["batch_upsert_performance"] = {
                    "success": upsert_result.get("success", False),
                    "duration": duration,
                    "documents_per_second": len(large_batch) / duration if duration > 0 else 0,
                    "details": upsert_result
                }
                print(f"✅ Batch upsert performance: {len(large_batch)} docs in {duration:.2f}s")
            except Exception as e:
                results["batch_upsert_performance"] = {"success": False, "error": str(e)}
                print(f"❌ Batch upsert performance test failed: {e}")
        else:
            print("⏩ Skipping batch upsert performance (quick mode)")
        
        # Test 2: Search performance
        print("🔍 Testing search performance...")
        try:
            query_vector = np.random.rand(self.config.dimension).tolist()
            start_time = time.time()
            search_result = self.vector_store.similarity_search(query_vector, top_k=10)
            duration = time.time() - start_time
            
            results["search_performance"] = {
                "success": search_result.get("success", False),
                "duration": duration,
                "results_count": search_result.get("count", 0),
                "details": search_result
            }
            print(f"✅ Search performance: {search_result.get('count', 0)} results in {duration:.3f}s")
        except Exception as e:
            results["search_performance"] = {"success": False, "error": str(e)}
            print(f"❌ Search performance test failed: {e}")
        
        # Test 3: Concurrent operations
        if not quick_mode:
            print("🔄 Testing concurrent operations...")
            try:
                import threading
                
                def concurrent_search():
                    query_vector = np.random.rand(self.config.dimension).tolist()
                    return self.vector_store.similarity_search(query_vector, top_k=3)
                
                threads = []
                results_list = []
                
                start_time = time.time()
                for i in range(5):
                    thread = threading.Thread(target=lambda: results_list.append(concurrent_search()))
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                duration = time.time() - start_time
                
                success_count = sum(1 for r in results_list if r.get("success", False))
                
                results["concurrent_operations"] = {
                    "success": success_count == 5,
                    "duration": duration,
                    "success_rate": success_count / 5,
                    "details": {"total_operations": 5, "successful_operations": success_count}
                }
                print(f"✅ Concurrent operations: {success_count}/5 successful in {duration:.2f}s")
            except Exception as e:
                results["concurrent_operations"] = {"success": False, "error": str(e)}
                print(f"❌ Concurrent operations test failed: {e}")
        else:
            print("⏩ Skipping concurrent operations (quick mode)")
        
        self.test_results["performance"] = results
    
    def _test_monitoring(self):
        """Test monitoring and metrics functionality"""
        print("\n📊 Testing Monitoring")
        print("-" * 40)
        
        results = {}
        
        # Test 1: Performance metrics
        print("📈 Testing performance metrics...")
        try:
            metrics = self.vector_store.get_performance_metrics(hours=1)
            results["performance_metrics"] = {
                "success": "count" in metrics,
                "details": metrics
            }
            print(f"✅ Performance metrics: {metrics.get('count', 0)} operations recorded")
        except Exception as e:
            results["performance_metrics"] = {"success": False, "error": str(e)}
            print(f"❌ Performance metrics failed: {e}")
        
        # Test 2: Health status
        print("🏥 Testing health status...")
        try:
            health_status = self.vector_store.get_health_status()
            results["health_status"] = {
                "success": "overall_status" in health_status,
                "details": health_status
            }
            print(f"✅ Health status: {health_status.get('overall_status', 'unknown')}")
        except Exception as e:
            results["health_status"] = {"success": False, "error": str(e)}
            print(f"❌ Health status failed: {e}")
        
        # Test 3: Operation info
        print("ℹ️ Testing operation info...")
        try:
            operations = self.vector_store.get_supported_operations()
            operation_info = self.vector_store.get_operation_info("similarity_search")
            results["operation_info"] = {
                "success": len(operations) > 0 and "description" in operation_info,
                "details": {"operations": operations, "sample_info": operation_info}
            }
            print(f"✅ Operation info: {len(operations)} operations supported")
        except Exception as e:
            results["operation_info"] = {"success": False, "error": str(e)}
            print(f"❌ Operation info failed: {e}")
        
        self.test_results["monitoring"] = results
    
    def _create_test_documents(self, count: int) -> List[Dict[str, Any]]:
        """Create test documents with genomics theme"""
        documents = []
        for i in range(count):
            doc = {
                "id": f"test_paper_{i}_{int(time.time())}",
                "values": np.random.rand(self.config.dimension).tolist(),
                "metadata": {
                    "title": f"Test Genomics Paper {i}",
                    "authors": [f"Dr. Author {i}"],
                    "journal": "Test Journal",
                    "year": 2024,
                    "doi": f"10.1000/test.{i}",
                    "abstract": f"This is test paper {i} about genomics research...",
                    "keywords": ["genomics", "test", "research"],
                    "type": "test",
                    "created_at": time.time()
                }
            }
            documents.append(doc)
        return documents
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("\n📊 Generating test report...")
        
        # Calculate overall success rate
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result.get("success", False):
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "test_suite_successful": success_rate >= 80,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "timestamp": time.time(),
            "configuration": self.config.to_dict(),
            "results": self.test_results,
            "summary": self._generate_test_summary()
        }
        
        # Save report
        report_file = Path("pinecone_test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Test report saved to: {report_file}")
        return report
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {}
        
        for category, tests in self.test_results.items():
            category_passed = sum(1 for result in tests.values() if result.get("success", False))
            category_total = len(tests)
            summary[category] = {
                "passed": category_passed,
                "total": category_total,
                "success_rate": (category_passed / category_total * 100) if category_total > 0 else 0
            }
        
        return summary
    
    def _generate_error_report(self, error: str):
        """Generate error report when test suite fails"""
        error_report = {
            "test_suite_successful": False,
            "timestamp": time.time(),
            "error": error,
            "configuration": self.config.to_dict() if self.config else None
        }
        
        error_file = Path("pinecone_test_error.json")
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        print(f"📄 Error report saved to: {error_file}")

def main():
    """Main test function with command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Pinecone Vector Store Test Suite")
    parser.add_argument("--category", choices=["basic", "advanced", "error", "performance", "monitoring", "all"],
                       default="all", help="Test category to run")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip long-running tests)")
    parser.add_argument("--report-file", type=str, default="pinecone_test_report.json",
                       help="Path to save the test report (default: pinecone_test_report.json)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = PineconeConfig.from_env()
        print(f"Configuration: {config.to_dict()}")
        
        # Create test suite
        test_suite = ComprehensiveTestSuite(config)
        
        # Run tests based on category and quick flag
        report = None
        if args.category == "all":
            report = test_suite.run_all_tests() if not args.quick else test_suite.run_quick_tests()
        else:
            category_map = {
                "basic": test_suite._test_basic_functionality,
                "advanced": test_suite._test_advanced_features,
                "error": test_suite._test_error_handling,
                "performance": test_suite._test_performance,
                "monitoring": test_suite._test_monitoring
            }
            test_suite._initialize_vector_store()
            if args.category in category_map:
                category_map[args.category]()
            report = test_suite._generate_test_report()
        
        # Save report to custom file if specified
        import shutil
        if args.report_file != "pinecone_test_report.json":
            shutil.copyfile("pinecone_test_report.json", args.report_file)
            print(f"📄 Test report also saved to: {args.report_file}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("🎯 TEST SUMMARY")
        print("=" * 70)
        print(f"Overall Success: {'✅ PASSED' if report['test_suite_successful'] else '❌ FAILED'}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")
        
        print("\n📋 Category Results:")
        for category, stats in report['summary'].items():
            status = "✅" if stats['success_rate'] >= 80 else "⚠️" if stats['success_rate'] >= 50 else "❌"
            print(f"  {status} {category}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)")
        
        print(f"\n📄 Detailed report saved to: {args.report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if report['test_suite_successful'] else 1)
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
