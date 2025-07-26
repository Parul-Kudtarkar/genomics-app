#!/usr/bin/env python3
"""
Quick test to determine which Pinecone import method works
"""

def test_pinecone_imports():
    print("Testing Pinecone import methods...")
    
    # Test 1: Try direct import (newest method)
    try:
        from pinecone import Pinecone
        print("✅ Method 1 works: from pinecone import Pinecone")
        
        # Try to create client (will fail with invalid key, but that's OK)
        try:
            pc = Pinecone(api_key="test")
            print("✅ Pinecone() constructor works")
        except Exception as e:
            if "api_key" in str(e).lower() or "invalid" in str(e).lower():
                print("✅ Pinecone() constructor works (invalid key error is expected)")
            else:
                print(f"❌ Pinecone() constructor failed: {e}")
        
        return "new"
    except ImportError as e:
        print(f"❌ Method 1 failed: {e}")
    except Exception as e:
        print(f"❌ Method 1 failed with unexpected error: {e}")
    
    # Test 2: Try module import with class access
    try:
        import pinecone
        if hasattr(pinecone, 'Pinecone'):
            print("✅ Method 2 works: import pinecone; pinecone.Pinecone()")
            return "new_alt"
        else:
            print("❌ Method 2: pinecone module has no Pinecone class")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Test 3: Try legacy method
    try:
        import pinecone
        if hasattr(pinecone, 'init'):
            print("✅ Method 3 works: import pinecone; pinecone.init()")
            return "legacy"
        else:
            print("❌ Method 3: pinecone module has no init function")
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    print("❌ All methods failed!")
    return None

if __name__ == "__main__":
    result = test_pinecone_imports()
    print(f"\nRecommended method: {result}")
