#!/usr/bin/env python3
"""
Debug script to see what content is actually being retrieved from the vector store
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Environment variables loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

from services.search_service import GenomicsSearchService
from services.rag_service import GenomicsRAGService

def debug_search_content():
    """Debug what content is being retrieved"""
    print("🔍 DEBUGGING RAG CONTENT RETRIEVAL")
    print("=" * 60)
    
    # Initialize services
    openai_api_key = os.getenv('OPENAI_API_KEY')
    search_service = GenomicsSearchService(openai_api_key=openai_api_key)
    rag_service = GenomicsRAGService(search_service=search_service, openai_api_key=openai_api_key)
    
    # Test questions
    test_questions = [
        "What is diabetes?",
        "What are islets?",
        "What is insulin?",
        "Explain difference between T1D and T2D",
        "What is scRNA?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        try:
            # Get raw search results
            print("🔍 Raw search results:")
            search_results = search_service.search_similar_chunks(question, top_k=8)
            
            for i, result in enumerate(search_results, 1):
                print(f"\n📄 Result {i}:")
                print(f"   Title: {result.get('title', 'Unknown')}")
                print(f"   Source: {result.get('source', 'Unknown')}")
                print(f"   Score: {result.get('score', 0):.3f}")
                print(f"   Content length: {len(result.get('content', ''))} chars")
                print(f"   Content preview: {result.get('content', '')[:200]}...")
                print(f"   Metadata: {result.get('metadata', {})}")
            
            # Get RAG response
            print(f"\n🤖 RAG Response:")
            rag_response = rag_service.ask_question(question, top_k=8)
            
            print(f"   Answer: {rag_response.answer[:300]}...")
            print(f"   Sources: {rag_response.num_sources}")
            print(f"   Confidence: {rag_response.confidence_score:.2f}")
            print(f"   Processing time: {rag_response.processing_time:.2f}s")
            
            # Show source details
            for i, source in enumerate(rag_response.sources[:2], 1):
                print(f"\n   📚 Source {i}:")
                print(f"      Title: {source.get('title', 'Unknown')}")
                print(f"      Journal: {source.get('journal', 'Unknown')}")
                print(f"      Year: {source.get('year', 'Unknown')}")
                print(f"      Relevance: {source.get('relevance_score', 0):.3f}")
                print(f"      Content preview: {source.get('content_preview', '')[:150]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_search_content() 