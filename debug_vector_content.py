#!/usr/bin/env python3
"""
Debug Vector Content
Examine what's actually stored in the vector database
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.search_service import GenomicsSearchService
from services.vector_store import PineconeVectorStore
from config.vector_db import PineconeConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def examine_vector_content():
    """Examine the actual content stored in vectors"""
    print("🔍 DEBUGGING VECTOR CONTENT")
    print("=" * 60)
    
    try:
        # Initialize services
        config = PineconeConfig.from_env()
        search_service = GenomicsSearchService(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        vector_store = PineconeVectorStore(config)
        
        print("✅ Services initialized")
        
        # Get some sample documents
        print("\n📚 EXAMINING SAMPLE DOCUMENTS")
        print("-" * 60)
        
        # Search for diabetes-related content
        results = search_service.search_similar_chunks(
            query_text="diabetes",
            top_k=5
        )
        
        print(f"Found {len(results)} documents")
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Document {i}:")
            print(f"   Title: {result.get('title', 'No title')}")
            print(f"   Journal: {result.get('journal', 'No journal')}")
            print(f"   Year: {result.get('year', 'No year')}")
            print(f"   Authors: {result.get('authors', [])}")
            print(f"   Content Length: {len(result.get('content', ''))} characters")
            print(f"   Content Preview: {result.get('content', '')[:200]}...")
            print(f"   Content Preview: {result.get('content_preview', '')[:200]}...")
            print(f"   Chunk Type: {result.get('chunk_type', 'Unknown')}")
            print(f"   Chunk Index: {result.get('chunk_index', 'Unknown')}")
            print(f"   Source File: {result.get('source_file', 'Unknown')}")
            print(f"   All Keys: {list(result.keys())}")
            
            # Check if content is actually empty
            content = result.get('content', '')
            if not content or content.strip() == '':
                print("   ❌ CONTENT IS EMPTY!")
            else:
                print(f"   ✅ Content available ({len(content)} chars)")
        
        # Check raw vector data
        print("\n🔍 EXAMINING RAW VECTOR DATA")
        print("-" * 60)
        
        # Get index stats
        stats = vector_store.get_index_stats()
        print(f"Total vectors: {stats.get('total_vector_count', 'Unknown')}")
        print(f"Dimension: {stats.get('dimension', 'Unknown')}")
        print(f"Namespaces: {stats.get('namespaces', {})}")
        
        # Try to get some raw vectors
        try:
            # This is a more direct approach to see what's stored
            print("\n🔍 CHECKING VECTOR METADATA")
            print("-" * 60)
            
            # Get a sample of vectors directly
            sample_query = "diabetes"
            sample_results = search_service.search_similar_chunks(
                query_text=sample_query,
                top_k=3
            )
            
            for i, result in enumerate(sample_results, 1):
                print(f"\nVector {i} Metadata:")
                for key, value in result.items():
                    if key != 'content':  # Skip content for readability
                        print(f"  {key}: {value}")
                
                # Show content separately
                content = result.get('content', '')
                print(f"  content: {'[EMPTY]' if not content else f'[{len(content)} chars] {content[:100]}...'}")
        
        except Exception as e:
            print(f"Error examining raw data: {e}")
        
        # Check if there's a content field issue
        print("\n🔍 CHECKING FIELD MAPPING")
        print("-" * 60)
        
        # Look for different possible content field names
        possible_content_fields = ['content', 'text', 'page_content', 'document_content', 'body']
        
        for i, result in enumerate(results[:2], 1):
            print(f"\nDocument {i} field analysis:")
            for field in possible_content_fields:
                value = result.get(field, 'NOT_FOUND')
                if value != 'NOT_FOUND':
                    print(f"  {field}: {len(str(value))} chars")
                else:
                    print(f"  {field}: NOT_FOUND")
        
    except Exception as e:
        logger.error(f"Error examining vector content: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    examine_vector_content() 