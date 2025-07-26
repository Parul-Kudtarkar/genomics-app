#!/usr/bin/env python3
"""
Test enhanced search capabilities of the vector store - FIXED VERSION
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Falling back to system environment variables...")

from pinecone import Pinecone
import openai

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
            print(f"❌ Failed to generate embedding: {e}")
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
            print(f"❌ Search failed: {e}")
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

def safe_join(items, separator=", ", max_items=None, default="N/A"):
    """Safely join a list of items, filtering out None values"""
    if not items:
        return default
    
    # Filter out None values and convert to strings
    safe_items = [str(item) for item in items if item is not None]
    
    if not safe_items:
        return default
    
    if max_items:
        safe_items = safe_items[:max_items]
    
    return separator.join(safe_items)

def print_search_results(results: List[Dict[str, Any]], title: str):
    """Print formatted search results - FIXED VERSION"""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        
        print(f"\n{i}. Score: {result['score']:.3f}")
        
        # Safe title handling
        title_text = metadata.get('title', 'N/A')
        if title_text and len(title_text) > 80:
            title_text = title_text[:80] + "..."
        print(f"   Title: {title_text}")
        
        # Safe journal handling
        journal = metadata.get('journal') or metadata.get('crossref_journal', 'N/A')
        print(f"   Journal: {journal}")
        
        # Safe year handling
        year = metadata.get('publication_year') or metadata.get('crossref_year', 'N/A')
        print(f"   Year: {year}")
        
        # Safe authors handling - this is where the original error occurred
        authors = metadata.get('authors', [])
        authors_text = safe_join(authors, max_items=3)
        print(f"   Authors: {authors_text}")
        
        # Safe citations handling
        citations = metadata.get('citation_count', 0)
        print(f"   Citations: {citations}")
        
        # Safe DOI handling
        if metadata.get('doi'):
            print(f"   DOI: {metadata['doi']}")
        
        # Safe chunk info handling
        chunk_type = metadata.get('chunk_type', 'N/A')
        chunk_index = metadata.get('chunk_index', 'N/A')
        print(f"   Chunk: {chunk_type} (index: {chunk_index})")

def test_basic_search(searcher: EnhancedVectorSearch):
    """Test basic text search"""
    print("🔍 Testing Basic Text Search")
    
    query = "CRISPR gene editing"
    results = searcher.search_by_text(query, top_k=5)
    print_search_results(results, f"Results for: '{query}'")

def test_filtered_searches(searcher: EnhancedVectorSearch):
    """Test various filtered searches"""
    
    # Test journal filter
    print("\n📰 Testing Journal Filter")
    results = searcher.search_by_text(
        query_text="machine learning",
        journal="Nature",
        top_k=3
    )
    print_search_results(results, "Machine learning papers in Nature")
    
    # Test year range filter
    print("\n📅 Testing Year Filter")
    results = searcher.search_by_text(
        query_text="RNA sequencing",
        year_range=(2020, 2024),
        top_k=3
    )
    print_search_results(results, "RNA sequencing papers (2020-2024)")
    
    # Test chunk type filter (abstracts only)
    print("\n📄 Testing Chunk Type Filter")
    results = searcher.search_abstracts_only("single cell analysis", top_k=3)
    print_search_results(results, "Single cell analysis (abstracts only)")
    
    # Test high-impact papers
    print("\n📊 Testing Citation Filter")
    results = searcher.get_high_impact_papers(min_citations=20, top_k=3)
    print_search_results(results, "High-impact papers (20+ citations)")

def test_specialized_searches(searcher: EnhancedVectorSearch):
    """Test specialized search functions"""
    
    # Test author search
    print("\n👥 Testing Author Search")
    # Get some authors from recent results first
    sample_results = searcher.search_by_text("genomics", top_k=10)
    
    authors_found = []
    for result in sample_results:
        authors = result['metadata'].get('authors', [])
        if authors:
            # Filter out None values when collecting authors
            valid_authors = [author for author in authors if author is not None]
            authors_found.extend(valid_authors[:2])  # Take first 2 valid authors
    
    if authors_found:
        test_author = authors_found[0]
        results = searcher.find_similar_papers_by_author(
            author_name=test_author,
            query_text="research",
            top_k=3
        )
        print_search_results(results, f"Papers by author: {test_author}")
    else:
        print("No authors found in sample data")
    
    # Test institution search
    print("\n🏛️ Testing Institution Search")
    # Get some institutions from recent results
    institutions_found = []
    for result in sample_results:
        institutions = result['metadata'].get('institutions', [])
        if institutions:
            # Filter out None values when collecting institutions
            valid_institutions = [inst for inst in institutions if inst is not None]
            institutions_found.extend(valid_institutions[:2])
    
    if institutions_found:
        test_institution = institutions_found[0]
        results = searcher.get_papers_by_institution(test_institution, top_k=3)
        print_search_results(results, f"Papers from: {test_institution}")
    else:
        print("No institutions found in sample data")

def test_keyword_search(searcher: EnhancedVectorSearch):
    """Test keyword-based filtering"""
    print("\n🔤 Testing Keyword Search")
    
    # Test with common scientific keywords
    keywords_to_test = ["CRISPR", "RNA", "protein", "cancer", "genomics"]
    
    for keyword in keywords_to_test:
        results = searcher.search_by_text(
            query_text="biological research",
            keywords=[keyword],
            top_k=2
        )
        
        if results:
            print(f"\nPapers with keyword '{keyword}':")
            for result in results[:2]:
                metadata = result['metadata']
                title = metadata.get('title', 'N/A')
                if title and len(title) > 60:
                    title = title[:60] + "..."
                journal = metadata.get('journal') or metadata.get('crossref_journal', 'N/A')
                print(f"  • {title} ({journal})")
            break
    else:
        print("No results found for tested keywords")

def run_search_demo():
    """Run comprehensive search demonstration"""
    try:
        print("🚀 Enhanced Vector Search Test Suite")
        print("=" * 50)
        
        # Initialize searcher
        searcher = EnhancedVectorSearch()
        
        # Get index info
        stats = searcher.index.describe_index_stats()
        print(f"Index: {searcher.index_name}")
        print(f"Total vectors: {stats.total_vector_count:,}")
        print(f"Dimension: {stats.dimension}")
        
        # Run tests
        test_basic_search(searcher)
        test_filtered_searches(searcher)
        test_specialized_searches(searcher)
        test_keyword_search(searcher)
        
        print(f"\n✅ Search testing completed!")
        print("\n💡 Try these search patterns in your application:")
        print("   - searcher.search_by_text('your query')")
        print("   - searcher.search_by_text('query', journal='Nature', year_range=(2020, 2024))")
        print("   - searcher.search_abstracts_only('your query')")
        print("   - searcher.get_high_impact_papers(min_citations=50)")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()

def interactive_search():
    """Interactive search mode"""
    try:
        searcher = EnhancedVectorSearch()
        
        print("🔍 Interactive Enhanced Search")
        print("=" * 40)
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Ask for filters
            print("\nOptional filters (press Enter to skip):")
            journal = input("Journal name: ").strip() or None
            author = input("Author name: ").strip() or None
            year_input = input("Year (YYYY) or range (YYYY-YYYY): ").strip()
            chunk_type = input("Chunk type (abstract/methods/results/discussion): ").strip() or None
            
            # Parse year input
            year_range = None
            if year_input:
                if '-' in year_input:
                    try:
                        start, end = year_input.split('-')
                        year_range = (int(start), int(end))
                    except:
                        print("Invalid year range format")
                else:
                    try:
                        year = int(year_input)
                        year_range = (year, year)
                    except:
                        print("Invalid year format")
            
            # Perform search
            print(f"\nSearching for: '{query}'...")
            results = searcher.search_by_text(
                query_text=query,
                journal=journal,
                author=author,
                year_range=year_range,
                chunk_type=chunk_type,
                top_k=5
            )
            
            print_search_results(results, f"Search Results")
    
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user")
    except Exception as e:
        print(f"❌ Interactive search failed: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced search capabilities')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run interactive search mode')
    parser.add_argument('--query', '-q', type=str, help='Run single query test')
    parser.add_argument('--journal', type=str, help='Filter by journal')
    parser.add_argument('--author', type=str, help='Filter by author')
    parser.add_argument('--year', type=str, help='Filter by year (YYYY or YYYY-YYYY)')
    parser.add_argument('--citations', type=int, help='Minimum citation count')
    parser.add_argument('--chunk-type', type=str, choices=['abstract', 'methods', 'results', 'discussion', 'content'], help='Filter by chunk type')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_search()
    elif args.query:
        # Single query mode
        try:
            searcher = EnhancedVectorSearch()
            
            # Parse year range
            year_range = None
            if args.year:
                if '-' in args.year:
                    start, end = args.year.split('-')
                    year_range = (int(start), int(end))
                else:
                    year = int(args.year)
                    year_range = (year, year)
            
            print(f"🔍 Searching for: '{args.query}'")
            if args.journal:
                print(f"   Journal filter: {args.journal}")
            if args.author:
                print(f"   Author filter: {args.author}")
            if year_range:
                print(f"   Year filter: {year_range[0]}-{year_range[1]}")
            if args.citations:
                print(f"   Min citations: {args.citations}")
            if args.chunk_type:
                print(f"   Chunk type: {args.chunk_type}")
            
            results = searcher.search_by_text(
                query_text=args.query,
                journal=args.journal,
                author=args.author,
                year_range=year_range,
                min_citations=args.citations,
                chunk_type=args.chunk_type,
                top_k=args.top_k
            )
            
            print_search_results(results, f"Search Results ({len(results)} found)")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
    else:
        # Default: run demo
        run_search_demo()

if __name__ == "__main__":
    main()
