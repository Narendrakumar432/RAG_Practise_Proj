import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("Testing GET /health ...")
    health_resp = requests.get(f"{BASE_URL}/health")
    print("Health response:", health_resp.status_code, health_resp.json())
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"

    queries = [
        {
            "query": "What are the constraints regarding agent frameworks and libraries?",
            "top_k": 3
        },
        {
            "query": "How should text chunking and overlap be designed and justified?",
            "top_k": 3
        },
        {
            "query": "What are the required deliverables and evaluation criteria for the assessment?",
            "top_k": 3
        }
    ]

    results_output = []

    print("\nExecuting test queries against /query ...\n")
    for q in queries:
        print(f"--- Query: '{q['query']}' (top_k={q['top_k']}) ---")
        resp = requests.post(f"{BASE_URL}/query", json=q)
        print("Status code:", resp.status_code)
        assert resp.status_code == 200, f"Query failed: {resp.text}"
        
        data = resp.json()
        print(f"Retrieved {len(data)} chunk(s):")
        for i, item in enumerate(data, 1):
            print(f"  [{i}] Page: {item['page_number']} | Score: {item['score']}")
            print(f"      Snippet: {item['chunk_text'][:120]}...\n")

        results_output.append({
            "query": q["query"],
            "top_k": q["top_k"],
            "results": data
        })

    # Save to results.json
    results_path = "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)

    print(f"\nSuccessfully wrote all query results to {results_path}!")

if __name__ == "__main__":
    test_api()
