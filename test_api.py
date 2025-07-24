# test_api.py

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status():
    print("📊 Testing status...")
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status working")
            print(f"   Vectors: {data['index_stats'].get('total_vectors', 'unknown')}")
            return True
        else:
            print(f"❌ Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status error: {e}")
        return False

def test_search():
    print("🔍 Testing vector search...")
    try:
        payload = {
            "query": "CRISPR gene editing",
            "top_k": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Search working")
            print(f"   Found {data['num_results']} results")
            print(f"   Time: {data['response_time_ms']}ms")
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_llm_query():
    print("🤖 Testing LLM query...")
    try:
        payload = {
            "query": "What is CRISPR?",
            "model": "gpt-3.5-turbo",  # Use cheaper model for testing
            "top_k": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ LLM query working")
            print(f"   Model: {data['model_used']}")
            print(f"   Sources: {data['num_sources']}")
            print(f"   Time: {data['response_time_ms']}ms")
            print(f"   Answer preview: {data['llm_response'][:100]}...")
            return True
        else:
            print(f"❌ LLM query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ LLM query error: {e}")
        return False

def test_filtered_search():
    print("🎯 Testing filtered search...")
    try:
        payload = {
            "query": "gene therapy",
            "top_k": 3,
            "year_start": 2020,
            "chunk_type": "abstract"
        }
        
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Filtered search working")
            print(f"   Results: {data['num_results']}")
            return True
        else:
            print(f"❌ Filtered search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Filtered search error: {e}")
        return False

def main():
    print("🧬 FastAPI Test Suite")
    print("=" * 50)
    print("Make sure your API is running: python main.py")
    print("=" * 50)
    
    tests = [
        test_health,
        test_status,
        test_search,
        test_llm_query,
        test_filtered_search
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 50)
    print("🎯 SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
    
    print("\\n💡 Example usage:")
    print('curl -X POST "http://localhost:8000/search" \\\\')
    print('  -H "Content-Type: application/json" \\\\')
    print('  -d \'{"query": "CRISPR", "top_k": 5}\'')

if __name__ == "__main__":
    main()
