import os
import requests
from pathlib import Path

BASE_URL = "http://localhost:10000"

def test_document_flow():
    # 1. Check initial state
    print("Checking initial state...")
    res = requests.get(f"{BASE_URL}/api/documents")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    initial_data = res.json()
    initial_chunks = initial_data["total_chunks"]
    print(f"Initial indexed chunks: {initial_chunks}")
    print(f"Initial documents: {[d['filename'] for d in initial_data['documents']]}\n")

    # 2. Create a test file
    test_filename = "test_upload_lifecycle.txt"
    test_content = "This is a temporary document to test the upload, duplicate prevention, and deletion functionality of the RAG system."
    test_path = Path("backend/data/banking_knowledge") / test_filename
    
    # Clean up if it was left over from a previous bad run
    if test_path.exists():
        os.remove(test_path)
    
    # We will upload it via the API, which will create the file
    print(f"Uploading new file: {test_filename}...")
    files = {"file": (test_filename, test_content, "text/plain")}
    res = requests.post(f"{BASE_URL}/api/upload", files=files)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    upload_data = res.json()
    print(f"Upload response: {upload_data}")
    
    # Check that file now exists on disk
    assert test_path.exists(), "Expected file to be created on disk"
    print("Verified file exists on disk.")

    # 3. Check doc list and chunk count
    res = requests.get(f"{BASE_URL}/api/documents")
    doc_list_data = res.json()
    print(f"Docs in list: {[d['filename'] for d in doc_list_data['documents']]}")
    print(f"New chunk count: {doc_list_data['total_chunks']}")
    assert test_filename in [d['filename'] for d in doc_list_data['documents']], "Expected test file in document list"
    assert doc_list_data['total_chunks'] > initial_chunks, "Expected chunk count to increase"
    
    # 4. Attempt duplicate upload
    print("\nAttempting duplicate upload...")
    files = {"file": (test_filename, test_content, "text/plain")}
    res = requests.post(f"{BASE_URL}/api/upload", files=files)
    print(f"Duplicate upload response status: {res.status_code}")
    print(f"Duplicate upload response: {res.text}")
    assert res.status_code == 400, f"Expected 400 duplicate error, got {res.status_code}"
    assert "already exists" in res.json()["detail"], "Expected duplicate error message"
    print("Verified duplicate upload is blocked successfully.")

    # 5. Delete document
    print(f"\nDeleting document {test_filename}...")
    res = requests.delete(f"{BASE_URL}/api/documents/{test_filename}")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    delete_data = res.json()
    print(f"Delete response: {delete_data}")
    
    # Check disk deletion
    assert not test_path.exists(), "Expected file to be deleted from disk"
    print("Verified file is deleted from disk.")

    # 6. Verify doc list and chunk count reverted
    res = requests.get(f"{BASE_URL}/api/documents")
    final_data = res.json()
    print(f"Final docs in list: {[d['filename'] for d in final_data['documents']]}")
    print(f"Final chunk count: {final_data['total_chunks']}")
    assert test_filename not in [d['filename'] for d in final_data['documents']], "Expected test file to be removed from document list"
    assert final_data['total_chunks'] == initial_chunks, f"Expected chunk count to revert to {initial_chunks}, got {final_data['total_chunks']}"
    
    print("\n--- ALL BACKEND LIFE CYCLE TESTS PASSED ---")

if __name__ == "__main__":
    test_document_flow()
