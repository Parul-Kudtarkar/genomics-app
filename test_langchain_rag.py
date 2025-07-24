#!/usr/bin/env python3
# test_langchain_rag.py - Test your LangChain RAG setup
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_langchain_imports():
    """Test LangChain imports"""
    print("🔍 Testing LangChain imports...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA
        from langchain.schema import Document
        from langchain.prompts import PromptTemplate
        print("✅ LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False

def test_core_dependencies():
    """Test core dependencies"""
    print("🔍 Testing core dependencies...")
    try:
        from pinecone import Pinecone
        from openai import OpenAI
        print("✅ Core dependencies successful")
        return True
    except ImportError as e:
        print(f"❌ Core dependency import failed: {e}")
        return False

def test_your_existing_services():
    """Test your existing services still work"""
    print("🔍 Testing your existing services...")
    try:
        from config.vector_db import PineconeConfig
        from services.vector_store import PineconeVectorStore
        from services.search_service import GenomicsSearchService
        
        # Test config
        config = PineconeConfig.from_env()
        print(f"✅ Config loaded: {config.index_name}")
        
        # Test search service
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            search_service = GenomicsSearchService(openai_api_key=openai_key)
            print("✅ Search service created")
        else:
            print("⚠️  OPENAI_API_KEY not set, skipping search service test")
        
        return True
    except Exception as e:
        print(f"❌ Existing services test failed: {e}")
        return False

def test_rag_service():
    """Test RAG service creation"""
    print("🔍 Testing RAG service...")
    try:
        from services.rag_service import create_rag_service
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("⚠️  OPENAI_API_KEY not set, skipping RAG test")
            return True
        
        # Create RAG service
        rag = create_rag_service(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo"  # Use cheaper model for testing
        )
        print("✅ RAG service created successfully")
        
        return True
    except Exception as e:
        print(f"❌ RAG service test failed: {e}")
        return False

def test_full_rag_pipeline():
    """Test full RAG pipeline with a question"""
    print("🔍 Testing full RAG pipeline...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    
    if not openai_key or not pinecone_key:
        print("⚠️  API keys not set, skipping full pipeline test")
        print("   Set OPENAI_API_KEY and PINECONE_API_KEY to test")
        return True
    
    try:
        from services.rag_service import create_rag_service
        
        # Create RAG service
        rag = create_rag_service(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo"
        )
        
        # Test with a simple question
        print("   Asking test question...")
        response = rag.ask_question("What is gene therapy?", top_k=2)
        
        print(f"✅ Full pipeline test successful!")
        print(f"   Answer length: {len(response['answer'])} characters")
        print(f"   Sources found: {response['num_sources']}")
        print(f"   Preview: {response['answer'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def print_installation_check():
    """Print installation verification"""
    print("\n📦 Package Version Check:")
    packages = [
        'langchain', 'langchain-openai', 'langchain-core', 
        'openai', 'pinecone-client', 'numpy'
    ]
    
    for package in packages:
        try:
            import importlib.metadata
            version = importlib.metadata.version(package)
            print(f"   {package}: {version}")
        except:
            print(f"   {package}: ❌ Not installed")

def main():
    """Run all tests"""
    print("🧬 LangChain RAG System Test")
    print("=" * 50)
    
    tests = [
        test_langchain_imports,
        test_core_dependencies,
        test_your_existing_services,
        test_rag_service,
        test_full_rag_pipeline
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Print summary
    print("=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Your LangChain RAG system is ready!")
        print("\n💡 Try this:")
        print("   from services.rag_service import create_rag_service")
        print("   rag = create_rag_service(os.getenv('OPENAI_API_KEY'))")
        print("   response = rag.ask_question('What is CRISPR?')")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        if passed < total:
            print("\n🔧 Next steps:")
            print("1. Run the installation commands again")
            print("2. Check your .env file has correct API keys")
            print("3. Verify your virtual environment is activated")
    
    print_installation_check()

if __name__ == "__main__":
    main()
