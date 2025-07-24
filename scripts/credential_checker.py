#!/usr/bin/env python3
"""
Pinecone Credential Checker and Index Setup
"""
import sys
import os
from pinecone import Pinecone, ServerlessSpec

def test_pinecone_credentials(api_key: str, index_name: str = None):
    """Test Pinecone credentials and setup"""
    
    print("🔑 Testing Pinecone credentials...")
    print(f"API Key starts with: {api_key[:10]}..." if len(api_key) > 10 else "API Key too short")
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client initialized successfully")
        
        # List available indexes
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        print(f"📊 Available indexes: {index_names}")
        
        if not index_names:
            print("⚠️ No indexes found. Let's create one.")
            
            if not index_name:
                index_name = input("Enter index name to create (e.g., 'genomics-publications'): ").strip()
            
            # Create index
            print(f"🏗️ Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Standard for text-embedding-ada-002
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            print(f"✅ Index '{index_name}' created successfully!")
            
            # Wait for it to be ready
            import time
            print("⏳ Waiting for index to be ready...")
            time.sleep(10)
            
        else:
            # Use existing index or specified one
            if index_name and index_name in index_names:
                print(f"✅ Index '{index_name}' found!")
            elif index_name and index_name not in index_names:
                print(f"❌ Index '{index_name}' not found!")
                print(f"Available indexes: {index_names}")
                index_name = index_names[0] if index_names else None
                print(f"Using '{index_name}' instead")
            else:
                index_name = index_names[0]
                print(f"Using first available index: '{index_name}'")
        
        if index_name:
            # Test index connection
            print(f"🔌 Testing connection to index '{index_name}'...")
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            print("✅ Index connection successful!")
            print(f"📊 Index stats:")
            print(f"   - Dimension: {stats.get('dimension', 'Unknown')}")
            print(f"   - Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"   - Index fullness: {stats.get('index_fullness', 0)}")
            
            return True, index_name
        else:
            print("❌ No usable index available")
            return False, None
            
    except Exception as e:
        print(f"❌ Pinecone test failed: {e}")
        
        # Common error hints
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n💡 Troubleshooting hints:")
            print("1. Check your API key is correct")
            print("2. Make sure you're using the right Pinecone account")
            print("3. Verify the API key has the right permissions")
            print("\n🔗 Get your API key from: https://app.pinecone.io/")
        
        return False, None

def test_openai_credentials(api_key: str):
    """Test OpenAI credentials"""
    
    print("\n🤖 Testing OpenAI credentials...")
    print(f"API Key starts with: {api_key[:10]}..." if len(api_key) > 10 else "API Key too short")
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple call
        models = client.models.list()
        print("✅ OpenAI client initialized successfully")
        
        # Test embedding generation
        response = client.embeddings.create(
            input=["test text"],
            model="text-embedding-ada-002"
        )
        
        embedding_dim = len(response.data[0].embedding)
        print(f"✅ Embedding generation works (dimension: {embedding_dim})")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n💡 Troubleshooting hints:")
            print("1. Check your OpenAI API key is correct")
            print("2. Make sure you have credits/billing set up")
            print("3. Verify the API key hasn't expired")
            print("\n🔗 Get your API key from: https://platform.openai.com/api-keys")
        
        return False

def main():
    """Main credential checker"""
    
    print("🔧 Pinecone & OpenAI Credential Checker")
    print("=" * 50)
    
    # Get credentials from command line or prompt
    if len(sys.argv) >= 3:
        openai_key = sys.argv[1]
        pinecone_key = sys.argv[2]
        index_name = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        print("Usage: python script.py <openai_key> <pinecone_key> [index_name]")
        print("Or run without args to enter interactively:")
        
        openai_key = input("\nEnter OpenAI API key: ").strip()
        pinecone_key = input("Enter Pinecone API key: ").strip()
        index_name = input("Enter Pinecone index name (optional): ").strip() or None
    
    # Test OpenAI
    openai_ok = test_openai_credentials(openai_key)
    
    # Test Pinecone
    pinecone_ok, final_index_name = test_pinecone_credentials(pinecone_key, index_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 CREDENTIAL TEST SUMMARY")
    print("=" * 50)
    print(f"OpenAI: {'✅ Working' if openai_ok else '❌ Failed'}")
    print(f"Pinecone: {'✅ Working' if pinecone_ok else '❌ Failed'}")
    if pinecone_ok:
        print(f"Index: {final_index_name}")
    
    if openai_ok and pinecone_ok:
        print("\n🎉 All credentials working! You can now run:")
        print(f"python simple_pdf_pipeline.py /path/to/pdfs {openai_key} {pinecone_key} {final_index_name}")
    else:
        print("\n❌ Fix the credential issues above before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
