import requests
import json

BASE_URL = "http://127.0.0.1:10000"

def test_upload():
    print("--- 1. Testing Document Ingestion (/api/upload) ---")
    file_path = "/Users/rishijaiswal/Downloads/Banking Support Chatbot/scratch/test_knowledge.txt"
    
    with open(file_path, 'rb') as f:
        files = {'file': ('test_knowledge.txt', f, 'text/plain')}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_chat():
    print("\n--- 2. Testing Semantic Retrieval & Response (/api/chat) ---")
    payload = {
        "message": "What is the interest rate of BankAssist Elite Plus?",
        "session_id": "test_session_id"
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response: {json.dumps(response_json, indent=2)}")
    
    print("\nSources Referenced:")
    for source in response_json.get("sources", []):
        print(f"- {source['source']} (Relevance: {round(source['relevance_score'] * 100)}%)")
        print(f"  Content: {source['content'][:120]}...")

if __name__ == "__main__":
    try:
        test_upload()
        test_chat()
    except Exception as e:
        print(f"Error testing integration: {e}")
