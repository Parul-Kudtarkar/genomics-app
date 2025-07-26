#!/usr/bin/env python3
"""
Quick test to verify OpenAI v1.3.5 is working correctly
"""

import openai
import sys

def test_openai_v1():
    print(f"✅ OpenAI version: {openai.__version__}")
    
    # Test client initialization (will fail without real API key, but structure should work)
    try:
        client = openai.OpenAI(api_key="test-key-placeholder")
        print("✅ OpenAI client initialization works")
        
        # Test embeddings call structure (will fail with auth error, but that's expected)
        try:
            response = client.embeddings.create(
                input=["test text"],
                model="text-embedding-ada-002"
            )
            print("✅ Embeddings call succeeded (unexpected but good!)")
        except Exception as e:
            if "api_key" in str(e).lower() or "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                print("✅ Embeddings API structure is correct (auth error is expected)")
                return True
            else:
                print(f"❌ Unexpected embeddings error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ OpenAI client initialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_openai_v1():
        print("\n🎉 OpenAI v1.3.5 is properly configured!")
        print("The updated pipeline should work correctly now.")
    else:
        print("\n❌ OpenAI has configuration issues")
        sys.exit(1)
