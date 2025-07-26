#!/usr/bin/env python3
"""
Check Vector Database Content
Query and display actual chunk content from Pinecone
"""
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_pinecone():
    """Initialize Pinecone client"""
    try:
        from pinecone import Pinecone
        api_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'genomics-publications')
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        return index
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone: {e}")
        return None

def get_index_stats(index):
    """Get index statistics"""
    try:
        stats = index.describe_index_stats()
        print(f"📊 Index Statistics:")
        print(f"   Total vectors: {stats.total_vector_count:,}")
        print(f"   Index dimension: {stats.dimension}")
        print(f"   Index name: {stats.index_name}")
        
        # Show namespace stats if available
        if hasattr(stats, 'namespaces') and stats.namespaces:
            print(f"   Namespaces: {list(stats.namespaces.keys())}")
        
        return stats
    except Exception as e:
        print(f"❌ Failed to get index stats: {e}")
        return None

def query_by_source(index, source: str, limit: int = 10):
    """Query vectors by source type"""
    try:
        print(f"\n🔍 Querying vectors with source='{source}' (limit={limit})...")
        
        # Query with metadata filter
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector for metadata-only query
            filter={"source": {"$eq": source}},
            top_k=limit,
            include_metadata=True,
            include_values=False  # Don't include vector values to save space
        )
        
        print(f"✅ Found {len(results.matches)} vectors")
        
        for i, match in enumerate(results.matches, 1):
            print(f"\n📄 Vector {i}:")
            print(f"   ID: {match.id}")
            print(f"   Score: {match.score:.4f}")
            
            metadata = match.metadata
            print(f"   Title: {metadata.get('title', 'N/A')[:100]}...")
            print(f"   Authors: {metadata.get('authors', 'N/A')}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            print(f"   Chunk Index: {metadata.get('chunk_index', 'N/A')}")
            print(f"   Word Count: {metadata.get('word_count', 'N/A')}")
            print(f"   Year: {metadata.get('year', 'N/A')}")
            print(f"   DOI: {metadata.get('doi', 'N/A')}")
            
            # Show file source
            if 'pdf_source' in metadata:
                print(f"   PDF Source: {metadata.get('pdf_source', 'N/A')}")
            elif 'xml_source' in metadata:
                print(f"   XML Source: {metadata.get('xml_source', 'N/A')}")
            
            # Show keywords
            keywords = metadata.get('keywords', '')
            if keywords:
                print(f"   Keywords: {keywords[:100]}...")
        
        return results.matches
        
    except Exception as e:
        print(f"❌ Failed to query by source: {e}")
        return []

def query_by_paper_id(index, paper_id: str):
    """Query all chunks for a specific paper"""
    try:
        print(f"\n🔍 Querying all chunks for paper_id='{paper_id}'...")
        
        results = index.query(
            vector=[0.0] * 1536,
            filter={"paper_id": {"$eq": paper_id}},
            top_k=50,  # Get all chunks for this paper
            include_metadata=True,
            include_values=False
        )
        
        print(f"✅ Found {len(results.matches)} chunks for paper {paper_id}")
        
        # Sort by chunk_index
        matches = sorted(results.matches, key=lambda x: x.metadata.get('chunk_index', 0))
        
        for i, match in enumerate(matches, 1):
            metadata = match.metadata
            print(f"\n📄 Chunk {metadata.get('chunk_index', i)}:")
            print(f"   ID: {match.id}")
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Word Count: {metadata.get('word_count', 'N/A')}")
            
            # Show a preview of the content (if available in metadata)
            # Note: The actual text content is not stored in metadata, only in the vector
            print(f"   Source: {metadata.get('source', 'N/A')}")
        
        return matches
        
    except Exception as e:
        print(f"❌ Failed to query by paper_id: {e}")
        return []

def search_by_text(index, query: str, limit: int = 5):
    """Search for vectors by text similarity"""
    try:
        print(f"\n🔍 Searching for: '{query}' (limit={limit})...")
        
        # You would need to generate embeddings for the query
        # For now, we'll use a simple metadata search
        print("⚠️  Text similarity search requires embedding generation")
        print("   Use Pinecone console for full-text search capabilities")
        
        return []
        
    except Exception as e:
        print(f"❌ Failed to search by text: {e}")
        return []

def list_all_sources(index):
    """List all unique sources in the database"""
    try:
        print(f"\n📋 Listing all sources in database...")
        
        # Get a sample of vectors to see what sources exist
        results = index.query(
            vector=[0.0] * 1536,
            top_k=1000,  # Get a large sample
            include_metadata=True,
            include_values=False
        )
        
        sources = set()
        for match in results.matches:
            source = match.metadata.get('source', 'unknown')
            sources.add(source)
        
        print(f"✅ Found sources: {list(sources)}")
        
        # Count vectors per source
        source_counts = {}
        for match in results.matches:
            source = match.metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\n📊 Vector counts by source:")
        for source, count in source_counts.items():
            print(f"   {source}: {count} vectors")
        
        return list(sources)
        
    except Exception as e:
        print(f"❌ Failed to list sources: {e}")
        return []

def main():
    """Main function"""
    print("🔍 Vector Database Content Checker")
    print("=" * 50)
    
    # Initialize Pinecone
    index = init_pinecone()
    if not index:
        sys.exit(1)
    
    # Get index stats
    stats = get_index_stats(index)
    if not stats:
        sys.exit(1)
    
    while True:
        print(f"\n📋 Available actions:")
        print(f"   1. List all sources")
        print(f"   2. Query PDF vectors")
        print(f"   3. Query XML vectors (PMC)")
        print(f"   4. Query by paper ID")
        print(f"   5. Search by text (placeholder)")
        print(f"   6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            list_all_sources(index)
            
        elif choice == '2':
            limit = input("Enter number of vectors to show (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            query_by_source(index, 'pdf', limit)
            
        elif choice == '3':
            limit = input("Enter number of vectors to show (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            query_by_source(index, 'pmc', limit)
            
        elif choice == '4':
            paper_id = input("Enter paper ID: ").strip()
            if paper_id:
                query_by_paper_id(index, paper_id)
            else:
                print("❌ Please enter a valid paper ID")
                
        elif choice == '5':
            query = input("Enter search query: ").strip()
            if query:
                search_by_text(index, query)
            else:
                print("❌ Please enter a valid query")
                
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main() 
