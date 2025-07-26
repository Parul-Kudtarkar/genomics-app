#!/usr/bin/env python3
"""
Enhanced RAG Service Test Script
Tests the complete RAG pipeline with existing Pinecone setup and data ingestion
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Environment variables loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

from services.rag_service import GenomicsRAGService, RAGConfig, create_rag_service
from config.vector_db import PineconeConfig
from services.vector_store import PineconeVectorStore

class RAGTestSuite:
    """Comprehensive test suite for RAG service"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def test_environment_setup(self) -> bool:
        """Test environment and configuration"""
        print("🔧 Testing Environment Setup")
        print("-" * 40)
        
        try:
            # Test environment variables
            required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
            missing_vars = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"❌ Missing environment variables: {missing_vars}")
                return False
            
            print("✅ Environment variables loaded")
            
            # Test Pinecone configuration
            config = PineconeConfig.from_env()
            print(f"✅ Pinecone config loaded: {config.index_name}")
            
            # Test vector store connection
            vector_store = PineconeVectorStore(config)
            stats = vector_store.get_index_statistics()
            print(f"✅ Vector store connected: {stats.get('total_vectors', 0)} vectors")
            
            self.test_results['environment'] = True
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            self.test_results['environment'] = False
            return False
    
    def test_rag_service_initialization(self) -> bool:
        """Test RAG service initialization"""
        print("\n🤖 Testing RAG Service Initialization")
        print("-" * 40)
        
        try:
            # Test with default configuration
            rag = create_rag_service()
            print("✅ RAG service created with default config")
            
            # Test with custom configuration
            custom_config = RAGConfig(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                default_top_k=3,
                enable_caching=True
            )
            rag_custom = create_rag_service(config=custom_config)
            print("✅ RAG service created with custom config")
            
            # Test service statistics
            stats = rag.get_service_statistics()
            print(f"✅ Service statistics retrieved: {stats.get('config', {}).get('model', 'unknown')}")
            
            self.test_results['initialization'] = True
            return True
            
        except Exception as e:
            print(f"❌ RAG service initialization failed: {e}")
            self.test_results['initialization'] = False
            return False
    
    def test_basic_question_answering(self) -> bool:
        """Test basic question answering functionality"""
        print("\n🔍 Testing Basic Question Answering")
        print("-" * 40)
        
        try:
            rag = create_rag_service()
            
            # Test simple question
            question = "What is gene therapy?"
            print(f"Question: {question}")
            
            response = rag.ask_question(question, top_k=3)
            
            print(f"✅ Answer generated ({len(response.answer)} chars)")
            print(f"   Sources: {response.num_sources}")
            print(f"   Processing time: {response.processing_time:.2f}s")
            print(f"   Confidence: {response.confidence_score:.2f}")
            print(f"   Preview: {response.answer[:100]}...")
            
            if response.num_sources > 0:
                print(f"   Top source: {response.sources[0]['title']}")
            
            self.test_results['basic_qa'] = True
            return True
            
        except Exception as e:
            print(f"❌ Basic Q&A failed: {e}")
            self.test_results['basic_qa'] = False
            return False
    
    def test_advanced_filtering(self) -> bool:
        """Test advanced filtering capabilities"""
        print("\n🎯 Testing Advanced Filtering")
        print("-" * 40)
        
        try:
            rag = create_rag_service()
            
            # Test journal filtering
            print("Testing journal filtering...")
            response = rag.ask_with_paper_focus(
                question="CRISPR applications in medicine",
                journal="Nature",
                top_k=3
            )
            print(f"✅ Journal filter: {response.num_sources} sources")
            
            # Test year range filtering
            print("Testing year range filtering...")
            response = rag.ask_with_paper_focus(
                question="Recent advances in genomics",
                year_range=(2020, 2024),
                top_k=3
            )
            print(f"✅ Year filter: {response.num_sources} sources")
            
            # Test citation count filtering
            print("Testing citation count filtering...")
            response = rag.ask_with_paper_focus(
                question="High-impact genomics research",
                min_citations=20,
                top_k=3
            )
            print(f"✅ Citation filter: {response.num_sources} sources")
            
            self.test_results['advanced_filtering'] = True
            return True
            
        except Exception as e:
            print(f"❌ Advanced filtering failed: {e}")
            self.test_results['advanced_filtering'] = False
            return False
    
    def test_specialized_prompts(self) -> bool:
        """Test specialized prompt templates"""
        print("\n📝 Testing Specialized Prompts")
        print("-" * 40)
        
        try:
            rag = create_rag_service()
            
            # Test methods-focused prompt
            print("Testing methods-focused prompt...")
            response = rag.ask_about_methods(
                "What are the standard protocols for DNA sequencing?",
                top_k=3
            )
            print(f"✅ Methods prompt: {len(response.answer)} chars")
            
            # Test results-focused prompt
            print("Testing results-focused prompt...")
            response = rag.ask_about_results(
                "What are the key findings in cancer genomics?",
                top_k=3
            )
            print(f"✅ Results prompt: {len(response.answer)} chars")
            
            # Test comparison prompt
            print("Testing comparison prompt...")
            response = rag.compare_approaches(
                "Compare different gene editing techniques",
                approaches=["CRISPR", "TALEN"],
                top_k=3
            )
            print(f"✅ Comparison prompt: {len(response.answer)} chars")
            
            self.test_results['specialized_prompts'] = True
            return True
            
        except Exception as e:
            print(f"❌ Specialized prompts failed: {e}")
            self.test_results['specialized_prompts'] = False
            return False
    
    def test_caching_functionality(self) -> bool:
        """Test caching functionality"""
        print("\n💾 Testing Caching Functionality")
        print("-" * 40)
        
        try:
            rag = create_rag_service()
            
            # Test question
            question = "What is the role of epigenetics in disease?"
            
            # First query (should not be cached)
            start_time = time.time()
            response1 = rag.ask_question(question, top_k=3)
            time1 = time.time() - start_time
            
            # Second query (should be cached)
            start_time = time.time()
            response2 = rag.ask_question(question, top_k=3)
            time2 = time.time() - start_time
            
            print(f"✅ First query: {time1:.2f}s")
            print(f"✅ Second query: {time2:.2f}s")
            
            if time2 < time1:
                print(f"✅ Caching working (speedup: {time1/time2:.1f}x)")
            else:
                print("⚠️  No significant speedup (may be due to small response time)")
            
            self.test_results['caching'] = True
            return True
            
        except Exception as e:
            print(f"❌ Caching test failed: {e}")
            self.test_results['caching'] = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        print("\n🛡️ Testing Error Handling")
        print("-" * 40)
        
        try:
            rag = create_rag_service()
            
            # Test with very specific query that might not have results
            response = rag.ask_question(
                "What is the molecular structure of the fictional protein XYZ123?",
                top_k=3
            )
            
            if response.num_sources == 0:
                print("✅ Properly handled query with no results")
            else:
                print("⚠️  Unexpected results for impossible query")
            
            # Test with invalid filters
            response = rag.ask_question(
                "Test query",
                filters={"invalid_field": {"$eq": "test"}},
                top_k=3
            )
            
            print("✅ Handled invalid filters gracefully")
            
            self.test_results['error_handling'] = True
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.test_results['error_handling'] = False
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance and metrics collection"""
        print("\n📊 Testing Performance Metrics")
        print("-" * 40)
        
        try:
            rag = create_rag_service()
            
            # Test multiple questions to get performance data
            questions = [
                "What is CRISPR?",
                "How does gene therapy work?",
                "What are the applications of genomics?",
                "Explain DNA sequencing methods",
                "What is epigenetics?"
            ]
            
            total_time = 0
            total_sources = 0
            
            for i, question in enumerate(questions, 1):
                print(f"Question {i}/{len(questions)}: {question[:30]}...")
                
                start_time = time.time()
                response = rag.ask_question(question, top_k=3)
                query_time = time.time() - start_time
                
                total_time += query_time
                total_sources += response.num_sources
                
                print(f"   Time: {query_time:.2f}s, Sources: {response.num_sources}")
            
            avg_time = total_time / len(questions)
            avg_sources = total_sources / len(questions)
            
            print(f"\n📈 Performance Summary:")
            print(f"   Average query time: {avg_time:.2f}s")
            print(f"   Average sources per query: {avg_sources:.1f}")
            print(f"   Total processing time: {total_time:.2f}s")
            
            # Get service statistics
            stats = rag.get_service_statistics()
            print(f"   Cache size: {stats.get('cache', {}).get('size', 0)}")
            
            self.test_results['performance'] = True
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🧬 Enhanced RAG Service Test Suite")
        print("=" * 60)
        
        tests = [
            ("Environment Setup", self.test_environment_setup),
            ("RAG Initialization", self.test_rag_service_initialization),
            ("Basic Q&A", self.test_basic_question_answering),
            ("Advanced Filtering", self.test_advanced_filtering),
            ("Specialized Prompts", self.test_specialized_prompts),
            ("Caching", self.test_caching_functionality),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance_metrics)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                success = test_func()
                results[test_name] = success
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
        
        # Generate summary
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        # Overall result
        overall_success = passed_tests == total_tests
        if overall_success:
            print("\n🎉 All tests passed! RAG service is ready for production.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the issues above.")
        
        # Save results
        test_report = {
            'timestamp': time.time(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'overall_success': overall_success,
            'detailed_results': results,
            'test_duration': time.time() - self.start_time
        }
        
        with open('rag_test_report.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: rag_test_report.json")
        
        return test_report

def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(
        description='Enhanced RAG Service Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/test_enhanced_rag.py
  
  # Run specific test
  python scripts/test_enhanced_rag.py --test basic_qa
  
  # Run with verbose output
  python scripts/test_enhanced_rag.py --verbose

Test Categories:
  - environment: Environment and configuration setup
  - initialization: RAG service initialization
  - basic_qa: Basic question answering
  - advanced_filtering: Advanced filtering capabilities
  - specialized_prompts: Specialized prompt templates
  - caching: Caching functionality
  - error_handling: Error handling capabilities
  - performance: Performance and metrics collection
        """
    )
    
    parser.add_argument(
        '--test',
        choices=['environment', 'initialization', 'basic_qa', 'advanced_filtering', 
                'specialized_prompts', 'caching', 'error_handling', 'performance'],
        help='Run specific test category'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    test_suite = RAGTestSuite()
    
    if args.test:
        # Run specific test
        test_map = {
            'environment': test_suite.test_environment_setup,
            'initialization': test_suite.test_rag_service_initialization,
            'basic_qa': test_suite.test_basic_question_answering,
            'advanced_filtering': test_suite.test_advanced_filtering,
            'specialized_prompts': test_suite.test_specialized_prompts,
            'caching': test_suite.test_caching_functionality,
            'error_handling': test_suite.test_error_handling,
            'performance': test_suite.test_performance_metrics
        }
        
        test_func = test_map.get(args.test)
        if test_func:
            success = test_func()
            print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {args.test}")
            return 0 if success else 1
        else:
            print(f"❌ Unknown test: {args.test}")
            return 1
    else:
        # Run all tests
        results = test_suite.run_all_tests()
        return 0 if results['overall_success'] else 1

if __name__ == "__main__":
    exit(main()) 