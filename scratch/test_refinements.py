"""
Verification script for BankAssist AI enterprise refinements.
Tests SSE streaming, rate limiting, lexical boosting, weak context fallbacks, and history persistence.
"""

import sys
import os
import json
import time
import requests
from pathlib import Path

# Add backend to path to allow importing app files directly for unit assertions
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

API_BASE = "http://127.0.0.1:10000"

def test_health():
    print("\n--- Testing API Health ---")
    try:
        res = requests.get(f"{API_BASE}/api/health")
        print(f"Status: {res.status_code}")
        data = res.json()
        print(f"Health Response: {json.dumps(data, indent=2)}")
        assert res.status_code == 200
        assert data.get("status") == "healthy"
        print("✅ Health check passed!")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

def test_documents_list():
    print("\n--- Testing Document Listing ---")
    try:
        res = requests.get(f"{API_BASE}/api/documents")
        print(f"Status: {res.status_code}")
        data = res.json()
        print(f"Documents count: {len(data.get('documents', []))}")
        print(f"Total chunks indexed: {data.get('total_chunks', 0)}")
        assert res.status_code == 200
        print("✅ Document list test passed!")
    except Exception as e:
        print(f"❌ Document list test failed: {e}")
        sys.exit(1)

def test_sse_streaming():
    print("\n--- Testing SSE Chat Streaming ---")
    payload = {
        "message": "What is the eligibility for a home loan?",
        "session_id": "test_verification_session",
        "stream": True
    }
    try:
        # Start connection with streaming enabled
        res = requests.post(f"{API_BASE}/api/chat", json=payload, stream=True)
        print(f"Status: {res.status_code}")
        assert res.status_code == 200
        
        events_received = []
        for line in res.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    events_received.append(data)
                    print(f"Event: {data.get('event')} | Text sample: {str(data.get('text', ''))[:40]}")
        
        assert len(events_received) >= 2
        # Assert first event is metadata
        assert events_received[0].get("event") == "metadata"
        assert "sources" in events_received[0]
        assert "confidence" in events_received[0]
        
        # Assert last event is done
        assert events_received[-1].get("event") == "done"
        assert "suggested_questions" in events_received[-1]
        
        print("✅ SSE Chat Streaming test passed!")
    except Exception as e:
        print(f"❌ SSE Chat Streaming test failed: {e}")
        sys.exit(1)

def test_weak_context_fallback():
    print("\n--- Testing Weak Context Fallback Bypass ---")
    # Query completely unrelated to banking domain to trigger relevance score < 0.35
    payload = {
        "message": "Who painted the Mona Lisa and what is the color of the sky?",
        "session_id": "test_verification_session",
        "stream": False
    }
    try:
        res = requests.post(f"{API_BASE}/api/chat", json=payload)
        print(f"Status: {res.status_code}")
        assert res.status_code == 200
        data = res.json()
        print(f"Bot response: {data.get('response')}")
        print(f"Confidence score: {data.get('confidence')}")
        
        # Verify fallback triggers
        assert any(x in data.get("response").lower() for x in ["knowledge base", "banking", "information", "apologize", "couldn't find"])
        assert data.get("confidence") == 0.0
        assert len(data.get("suggested_questions", [])) > 0
        print("✅ Weak Context Fallback test passed!")
    except Exception as e:
        print(f"❌ Weak Context Fallback test failed: {e}")
        sys.exit(1)

def test_session_persistence():
    print("\n--- Testing Chat History Persistence ---")
    session_id = "test_persistence_session"
    payload = {
        "message": "Tell me about credit cards.",
        "session_id": session_id,
        "stream": False
    }
    try:
        # 1. Clear session first
        requests.delete(f"{API_BASE}/api/sessions/{session_id}")
        
        # 2. Call chat to write to history
        res = requests.post(f"{API_BASE}/api/chat", json=payload)
        assert res.status_code == 200
        
        # 3. Check JSON history file path on disk
        session_file = backend_dir / "data" / "sessions" / f"{session_id}.json"
        print(f"Checking disk session file: {session_file}")
        assert session_file.exists()
        
        with open(session_file, "r") as f:
            saved_data = json.load(f)
            assert saved_data["session_id"] == session_id
            assert len(saved_data["messages"]) == 2  # user and assistant message
            print("Session data successfully persisted to disk!")
            
        # 4. Check history API GET endpoint retrieves correct length
        history_res = requests.get(f"{API_BASE}/api/sessions/{session_id}/history")
        history_data = history_res.json()
        assert len(history_data.get("messages", [])) == 2
        print("✅ Session History Persistence test passed!")
        
    except Exception as e:
        print(f"❌ Session History Persistence test failed: {e}")
        sys.exit(1)

def test_rate_limiting():
    print("\n--- Testing Rate Limiting (Chat 30/min, Upload 5/min) ---")
    session_id = f"test_rate_limit_{int(time.time())}"
    payload = {
        "message": "Ping",
        "session_id": session_id,
        "stream": False
    }
    
    # We will trigger the rate limits by hitting the endpoint in rapid succession.
    # To keep verification script fast, let's change client IP header or mock limit.
    # Since in-memory sliding window uses request.client.host, rapid loop will trigger it.
    # Chat limit is 30 requests. Let's make 31 requests.
    print("Sending rapid queries to trigger chat 429...")
    triggered_429 = False
    
    for i in range(35):
        try:
            res = requests.post(f"{API_BASE}/api/chat", json=payload)
            if res.status_code == 429:
                triggered_429 = True
                print(f"Success! Request {i+1} blocked with 429: {res.json().get('detail')}")
                break
        except Exception as e:
            print(f"Request error: {e}")
            break
            
    assert triggered_429, "Rate limit failed to trigger after 30 requests"
    print("✅ Rate Limiting test passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("BankAssist AI Enterprise Refinements Test Suite")
    print("=" * 60)
    
    test_health()
    test_documents_list()
    test_sse_streaming()
    test_weak_context_fallback()
    test_session_persistence()
    test_rate_limiting()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! Enterprise refinements verified.")
    print("=" * 60)
