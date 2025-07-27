#!/usr/bin/env python3
"""
Comprehensive Vector Store Explorer
Allows you to explore everything in your genomics vector store
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.search_service import GenomicsSearchService
from services.vector_store import PineconeVectorStore
from config.vector_db import PineconeConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreExplorer:
    """Comprehensive explorer for the genomics vector store"""
    
    def __init__(self):
        """Initialize the explorer with search and vector store services"""
        try:
            # Load configuration
            self.config = PineconeConfig.from_env()
            
            # Initialize services
            self.search_service = GenomicsSearchService(
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            self.vector_store = PineconeVectorStore(self.config)
            
            logger.info("✅ Vector store explorer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize explorer: {e}")
            raise
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store"""
        print("\n📊 VECTOR STORE STATISTICS")
        print("=" * 50)
        
        try:
            # Get basic stats
            stats = self.vector_store.get_index_stats()
            print(f"Total Vectors: {stats.get('total_vector_count', 'Unknown')}")
            print(f"Index Dimension: {stats.get('dimension', 'Unknown')}")
            print(f"Index Type: {stats.get('index_type', 'Unknown')}")
            print(f"Metric: {stats.get('metric', 'Unknown')}")
            
            # Get search service stats
            search_stats = self.search_service.get_search_statistics()
            print(f"Search Service Stats: {search_stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def explore_by_journal(self, journal_name: str = None, limit: int = 50):
        """Explore documents by journal"""
        print(f"\n📰 JOURNAL EXPLORATION")
        print("=" * 50)
        
        if journal_name:
            print(f"Exploring journal: {journal_name}")
            filters = {"journal": journal_name}
        else:
            print("Exploring all journals")
            filters = {}
        
        try:
            # Get documents from this journal
            results = self.search_service.search_similar_chunks(
                query_text="",  # Empty query to get all
                top_k=limit,
                filters=filters
            )
            
            print(f"Found {len(results)} documents")
            
            # Group by journal if no specific journal
            if not journal_name:
                journals = {}
                for result in results:
                    journal = result.get('journal', 'Unknown')
                    if journal not in journals:
                        journals[journal] = []
                    journals[journal].append(result)
                
                print(f"\nJournals found ({len(journals)}):")
                for journal, docs in sorted(journals.items()):
                    print(f"  {journal}: {len(docs)} documents")
            
            # Show sample documents
            print(f"\nSample documents:")
            for i, result in enumerate(results[:10], 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Journal: {result.get('journal', 'Unknown')}")
                print(f"   Year: {result.get('year', 'Unknown')}")
                print(f"   Authors: {', '.join(result.get('authors', [])[:3])}")
                print(f"   Citations: {result.get('citation_count', 0)}")
                print()
            
        except Exception as e:
            logger.error(f"Error exploring by journal: {e}")
    
    def explore_by_year(self, start_year: int = None, end_year: int = None, limit: int = 50):
        """Explore documents by publication year"""
        print(f"\n📅 YEAR-BASED EXPLORATION")
        print("=" * 50)
        
        if start_year and end_year:
            print(f"Exploring years {start_year}-{end_year}")
            filters = {"publication_year": {"$gte": start_year, "$lte": end_year}}
        elif start_year:
            print(f"Exploring from year {start_year} onwards")
            filters = {"publication_year": {"$gte": start_year}}
        elif end_year:
            print(f"Exploring up to year {end_year}")
            filters = {"publication_year": {"$lte": end_year}}
        else:
            print("Exploring all years")
            filters = {}
        
        try:
            results = self.search_service.search_similar_chunks(
                query_text="",
                top_k=limit,
                filters=filters
            )
            
            print(f"Found {len(results)} documents")
            
            # Group by year
            years = {}
            for result in results:
                year = result.get('year', 'Unknown')
                if year not in years:
                    years[year] = []
                years[year].append(result)
            
            print(f"\nDocuments by year:")
            for year in sorted(years.keys(), reverse=True):
                print(f"  {year}: {len(years[year])} documents")
            
            # Show recent documents
            print(f"\nRecent documents:")
            for i, result in enumerate(results[:10], 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Year: {result.get('year', 'Unknown')}")
                print(f"   Journal: {result.get('journal', 'Unknown')}")
                print(f"   Citations: {result.get('citation_count', 0)}")
                print()
            
        except Exception as e:
            logger.error(f"Error exploring by year: {e}")
    
    def explore_by_author(self, author_name: str = None, limit: int = 50):
        """Explore documents by author"""
        print(f"\n👤 AUTHOR EXPLORATION")
        print("=" * 50)
        
        if author_name:
            print(f"Exploring author: {author_name}")
            filters = {"authors": {"$in": [author_name]}}
        else:
            print("Exploring all authors")
            filters = {}
        
        try:
            results = self.search_service.search_similar_chunks(
                query_text="",
                top_k=limit,
                filters=filters
            )
            
            print(f"Found {len(results)} documents")
            
            # Group by author if no specific author
            if not author_name:
                authors = {}
                for result in results:
                    for author in result.get('authors', []):
                        if author not in authors:
                            authors[author] = []
                        authors[author].append(result)
                
                print(f"\nTop authors found:")
                sorted_authors = sorted(authors.items(), key=lambda x: len(x[1]), reverse=True)
                for author, docs in sorted_authors[:10]:
                    print(f"  {author}: {len(docs)} documents")
            
            # Show sample documents
            print(f"\nSample documents:")
            for i, result in enumerate(results[:10], 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Authors: {', '.join(result.get('authors', [])[:3])}")
                print(f"   Journal: {result.get('journal', 'Unknown')}")
                print(f"   Year: {result.get('year', 'Unknown')}")
                print()
            
        except Exception as e:
            logger.error(f"Error exploring by author: {e}")
    
    def explore_by_citation_count(self, min_citations: int = 0, max_citations: int = None, limit: int = 50):
        """Explore documents by citation count"""
        print(f"\n📈 CITATION-BASED EXPLORATION")
        print("=" * 50)
        
        if max_citations:
            print(f"Exploring documents with {min_citations}-{max_citations} citations")
            filters = {"citation_count": {"$gte": min_citations, "$lte": max_citations}}
        else:
            print(f"Exploring documents with {min_citations}+ citations")
            filters = {"citation_count": {"$gte": min_citations}}
        
        try:
            results = self.search_service.search_similar_chunks(
                query_text="",
                top_k=limit,
                filters=filters
            )
            
            print(f"Found {len(results)} documents")
            
            # Sort by citation count
            sorted_results = sorted(results, key=lambda x: x.get('citation_count', 0), reverse=True)
            
            print(f"\nTop cited documents:")
            for i, result in enumerate(sorted_results[:15], 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Citations: {result.get('citation_count', 0)}")
                print(f"   Journal: {result.get('journal', 'Unknown')}")
                print(f"   Year: {result.get('year', 'Unknown')}")
                print(f"   Authors: {', '.join(result.get('authors', [])[:3])}")
                print()
            
        except Exception as e:
            logger.error(f"Error exploring by citation count: {e}")
    
    def explore_by_topic(self, topic: str, limit: int = 50):
        """Explore documents by topic/keyword"""
        print(f"\n🔍 TOPIC EXPLORATION")
        print("=" * 50)
        print(f"Exploring topic: {topic}")
        
        try:
            results = self.search_service.search_similar_chunks(
                query_text=topic,
                top_k=limit
            )
            
            print(f"Found {len(results)} documents related to '{topic}'")
            
            # Show results with relevance scores
            print(f"\nTop documents for '{topic}':")
            for i, result in enumerate(results[:15], 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Relevance Score: {result.get('relevance_score', 0):.3f}")
                print(f"   Journal: {result.get('journal', 'Unknown')}")
                print(f"   Year: {result.get('year', 'Unknown')}")
                print(f"   Citations: {result.get('citation_count', 0)}")
                print(f"   Content Preview: {result.get('content_preview', '')[:100]}...")
                print()
            
        except Exception as e:
            logger.error(f"Error exploring by topic: {e}")
    
    def get_random_sample(self, sample_size: int = 20):
        """Get a random sample of documents"""
        print(f"\n🎲 RANDOM SAMPLE EXPLORATION")
        print("=" * 50)
        print(f"Getting random sample of {sample_size} documents")
        
        try:
            # Use a generic query to get diverse results
            results = self.search_service.search_similar_chunks(
                query_text="genomics research",
                top_k=sample_size
            )
            
            print(f"Sample documents:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.get('title', 'No title')}")
                print(f"   Journal: {result.get('journal', 'Unknown')}")
                print(f"   Year: {result.get('year', 'Unknown')}")
                print(f"   Authors: {', '.join(result.get('authors', [])[:3])}")
                print(f"   Citations: {result.get('citation_count', 0)}")
                print(f"   Content Type: {result.get('chunk_type', 'Unknown')}")
                print()
            
        except Exception as e:
            logger.error(f"Error getting random sample: {e}")
    
    def comprehensive_analysis(self):
        """Run a comprehensive analysis of the entire vector store"""
        print("\n🔬 COMPREHENSIVE VECTOR STORE ANALYSIS")
        print("=" * 60)
        
        # Get statistics
        self.get_store_statistics()
        
        # Explore by different dimensions
        print("\n" + "=" * 60)
        self.explore_by_journal(limit=30)
        
        print("\n" + "=" * 60)
        self.explore_by_year(start_year=2020, limit=30)
        
        print("\n" + "=" * 60)
        self.explore_by_citation_count(min_citations=10, limit=30)
        
        print("\n" + "=" * 60)
        self.get_random_sample(sample_size=15)
        
        print("\n" + "=" * 60)
        print("🎉 Comprehensive analysis completed!")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Comprehensive Vector Store Explorer")
    parser.add_argument("--mode", choices=[
        "stats", "journal", "year", "author", "citations", 
        "topic", "random", "comprehensive"
    ], default="comprehensive", help="Exploration mode")
    
    parser.add_argument("--journal", type=str, help="Journal name to explore")
    parser.add_argument("--start-year", type=int, help="Start year for exploration")
    parser.add_argument("--end-year", type=int, help="End year for exploration")
    parser.add_argument("--author", type=str, help="Author name to explore")
    parser.add_argument("--min-citations", type=int, default=0, help="Minimum citation count")
    parser.add_argument("--max-citations", type=int, help="Maximum citation count")
    parser.add_argument("--topic", type=str, help="Topic/keyword to explore")
    parser.add_argument("--limit", type=int, default=50, help="Number of documents to retrieve")
    parser.add_argument("--sample-size", type=int, default=20, help="Random sample size")
    
    args = parser.parse_args()
    
    try:
        explorer = VectorStoreExplorer()
        
        if args.mode == "stats":
            explorer.get_store_statistics()
        elif args.mode == "journal":
            explorer.explore_by_journal(args.journal, args.limit)
        elif args.mode == "year":
            explorer.explore_by_year(args.start_year, args.end_year, args.limit)
        elif args.mode == "author":
            explorer.explore_by_author(args.author, args.limit)
        elif args.mode == "citations":
            explorer.explore_by_citation_count(args.min_citations, args.max_citations, args.limit)
        elif args.mode == "topic":
            if not args.topic:
                print("❌ Please provide a topic with --topic")
                return
            explorer.explore_by_topic(args.topic, args.limit)
        elif args.mode == "random":
            explorer.get_random_sample(args.sample_size)
        elif args.mode == "comprehensive":
            explorer.comprehensive_analysis()
        
    except Exception as e:
        logger.error(f"❌ Exploration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 