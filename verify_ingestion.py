#!/usr/bin/env python3
"""
Verify Document Ingestion Quality
================================

This script comprehensively tests that all documents have been ingested correctly
with proper content, metadata, and functionality.
"""

import os
import sys
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.search_service import GenomicsSearchService
from services.vector_store import PineconeVectorStore
from config.vector_db import PineconeConfig

@dataclass
class IngestionTestResult:
    """Results from ingestion verification tests"""
    test_name: str
    passed: bool
    details: str
    metrics: Dict[str, Any] = None

class IngestionVerifier:
    """Comprehensive ingestion verification tool"""
    
    def __init__(self):
        """Initialize the verifier with search and vector store services"""
        try:
            # Load configuration
            config = PineconeConfig.from_env()
            
            # Initialize services with proper API keys
            self.search_service = GenomicsSearchService(
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            self.vector_store = PineconeVectorStore(config)
            print("✅ Services initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            sys.exit(1)
    
    def test_vector_store_stats(self) -> IngestionTestResult:
        """Test basic vector store statistics"""
        try:
            stats = self.vector_store.get_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            # Debug: Print the full stats to see what we're getting
            print(f"DEBUG: Full stats from vector store: {stats}")
            
            if total_vectors == 0:
                return IngestionTestResult(
                    test_name="Vector Store Statistics",
                    passed=False,
                    details="No vectors found in the store",
                    metrics={"total_vectors": 0}
                )
            
            return IngestionTestResult(
                test_name="Vector Store Statistics",
                passed=True,
                details=f"Found {total_vectors} vectors in the store",
                metrics={
                    "total_vectors": total_vectors,
                    "dimension": stats.get('dimension'),
                    "index_fullness": stats.get('index_fullness'),
                    "namespaces": stats.get('namespaces')
                }
            )
        except Exception as e:
            return IngestionTestResult(
                test_name="Vector Store Statistics",
                passed=False,
                details=f"Error getting stats: {e}"
            )
    
    def test_content_quality(self, sample_size: int = 10) -> IngestionTestResult:
        """Test that documents have proper content (not empty)"""
        try:
            # Get random sample
            results = self.search_service.search_similar_chunks(
                query_text="genomics research",
                top_k=sample_size
            )
            
            empty_content = 0
            total_content_length = 0
            content_lengths = []
            
            for result in results:
                content = result.get('content', '')
                content_length = len(content)
                content_lengths.append(content_length)
                total_content_length += content_length
                
                if not content.strip():
                    empty_content += 1
            
            avg_content_length = total_content_length / len(results) if results else 0
            
            if empty_content > 0:
                return IngestionTestResult(
                    test_name="Content Quality",
                    passed=False,
                    details=f"{empty_content}/{len(results)} documents have empty content",
                    metrics={
                        "sample_size": sample_size,
                        "empty_content": empty_content,
                        "avg_content_length": avg_content_length,
                        "content_lengths": content_lengths
                    }
                )
            
            return IngestionTestResult(
                test_name="Content Quality",
                passed=True,
                details=f"All {len(results)} sampled documents have content (avg: {avg_content_length:.0f} chars)",
                metrics={
                    "sample_size": sample_size,
                    "empty_content": 0,
                    "avg_content_length": avg_content_length,
                    "content_lengths": content_lengths
                }
            )
        except Exception as e:
            return IngestionTestResult(
                test_name="Content Quality",
                passed=False,
                details=f"Error testing content: {e}"
            )
    
    def test_metadata_completeness(self, sample_size: int = 10) -> IngestionTestResult:
        """Test that documents have complete metadata"""
        try:
            results = self.search_service.search_similar_chunks(
                query_text="genomics research",
                top_k=sample_size
            )
            
            missing_fields = {
                'title': 0,
                'journal': 0,
                'year': 0,
                'authors': 0,
                'doi': 0
            }
            
            total_docs = len(results)
            
            for result in results:
                metadata = result.get('metadata', {})
                
                if not metadata.get('title'):
                    missing_fields['title'] += 1
                if not metadata.get('journal'):
                    missing_fields['journal'] += 1
                if not metadata.get('year'):
                    missing_fields['year'] += 1
                if not metadata.get('authors'):
                    missing_fields['authors'] += 1
                if not metadata.get('doi'):
                    missing_fields['doi'] += 1
            
            # Calculate completeness percentage
            total_fields = len(missing_fields) * total_docs
            missing_total = sum(missing_fields.values())
            completeness = ((total_fields - missing_total) / total_fields) * 100
            
            if completeness < 80:  # Less than 80% complete
                return IngestionTestResult(
                    test_name="Metadata Completeness",
                    passed=False,
                    details=f"Only {completeness:.1f}% metadata completeness",
                    metrics={
                        "completeness_percentage": completeness,
                        "missing_fields": missing_fields,
                        "sample_size": sample_size
                    }
                )
            
            return IngestionTestResult(
                test_name="Metadata Completeness",
                passed=True,
                details=f"Metadata is {completeness:.1f}% complete",
                metrics={
                    "completeness_percentage": completeness,
                    "missing_fields": missing_fields,
                    "sample_size": sample_size
                }
            )
        except Exception as e:
            return IngestionTestResult(
                test_name="Metadata Completeness",
                passed=False,
                details=f"Error testing metadata: {e}"
            )
    
    def test_search_functionality(self) -> IngestionTestResult:
        """Test that search functionality works correctly"""
        try:
            # Test different types of queries
            test_queries = [
                "diabetes",
                "CRISPR",
                "gene therapy",
                "cancer research"
            ]
            
            search_results = {}
            total_results = 0
            
            for query in test_queries:
                results = self.search_service.search_similar_chunks(
                    query_text=query,
                    top_k=5
                )
                search_results[query] = len(results)
                total_results += len(results)
            
            avg_results = total_results / len(test_queries)
            
            if avg_results < 2:  # Less than 2 results per query on average
                return IngestionTestResult(
                    test_name="Search Functionality",
                    passed=False,
                    details=f"Low search results: {avg_results:.1f} average per query",
                    metrics={
                        "avg_results_per_query": avg_results,
                        "search_results": search_results
                    }
                )
            
            return IngestionTestResult(
                test_name="Search Functionality",
                passed=True,
                details=f"Search working well: {avg_results:.1f} average results per query",
                metrics={
                    "avg_results_per_query": avg_results,
                    "search_results": search_results
                }
            )
        except Exception as e:
            return IngestionTestResult(
                test_name="Search Functionality",
                passed=False,
                details=f"Error testing search: {e}"
            )
    
    def test_document_diversity(self) -> IngestionTestResult:
        """Test that you have diverse documents (not all the same)"""
        try:
            results = self.search_service.search_similar_chunks(
                query_text="genomics research",
                top_k=20
            )
            
            unique_titles = set()
            unique_journals = set()
            unique_years = set()
            
            for result in results:
                metadata = result.get('metadata', {})
                unique_titles.add(metadata.get('title', 'Unknown'))
                unique_journals.add(metadata.get('journal', 'Unknown'))
                unique_years.add(metadata.get('year', 'Unknown'))
            
            diversity_score = (len(unique_titles) + len(unique_journals) + len(unique_years)) / 3
            
            if diversity_score < 3:  # Less than 3 unique items on average
                return IngestionTestResult(
                    test_name="Document Diversity",
                    passed=False,
                    details=f"Low diversity: {diversity_score:.1f} average unique items",
                    metrics={
                        "diversity_score": diversity_score,
                        "unique_titles": len(unique_titles),
                        "unique_journals": len(unique_journals),
                        "unique_years": len(unique_years)
                    }
                )
            
            return IngestionTestResult(
                test_name="Document Diversity",
                passed=True,
                details=f"Good diversity: {diversity_score:.1f} average unique items",
                metrics={
                    "diversity_score": diversity_score,
                    "unique_titles": len(unique_titles),
                    "unique_journals": len(unique_journals),
                    "unique_years": len(unique_years)
                }
            )
        except Exception as e:
            return IngestionTestResult(
                test_name="Document Diversity",
                passed=False,
                details=f"Error testing diversity: {e}"
            )
    
    def run_all_tests(self) -> List[IngestionTestResult]:
        """Run all verification tests"""
        print("🔍 VERIFYING DOCUMENT INGESTION")
        print("=" * 50)
        
        tests = [
            self.test_vector_store_stats,
            self.test_content_quality,
            self.test_metadata_completeness,
            self.test_search_functionality,
            self.test_document_diversity
        ]
        
        results = []
        
        for test in tests:
            print(f"\n🧪 Running: {test.__name__}")
            result = test()
            results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {status}: {result.details}")
            
            if result.metrics:
                for key, value in result.metrics.items():
                    print(f"   📊 {key}: {value}")
        
        return results
    
    def generate_report(self, results: List[IngestionTestResult]) -> str:
        """Generate a comprehensive test report"""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        report = f"""
📋 INGESTION VERIFICATION REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 SUMMARY
{'=' * 20}
Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)
Overall Status: {'✅ PASSED' if passed == total else '❌ FAILED'}

📝 DETAILED RESULTS
{'=' * 20}
"""
        
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report += f"\n{status} {result.test_name}"
            report += f"\n   Details: {result.details}"
            
            if result.metrics:
                report += "\n   Metrics:"
                for key, value in result.metrics.items():
                    report += f"\n     {key}: {value}"
            report += "\n"
        
        return report

def main():
    """Main verification function"""
    verifier = IngestionVerifier()
    results = verifier.run_all_tests()
    
    # Generate and display report
    report = verifier.generate_report(results)
    print("\n" + report)
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"ingestion_verification_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Exit with appropriate code
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    if passed == total:
        print("\n🎉 All tests passed! Your documents are ingested correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the report for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 