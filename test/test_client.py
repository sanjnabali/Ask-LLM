# test_client.py
import requests
import json
import asyncio
import aiohttp

# Configuration
API_BASE_URL = "https://your-api-url.com/api/v1"  # Replace with your deployed URL
# For local testing: "http://localhost:8000/api/v1"
AUTH_TOKEN = "4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health")
        print("Health Check Response:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_hackrx_endpoint():
    """Test the main hackrx endpoint"""
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "46-year-old male, knee surgery in Pune, 3-month-old insurance policy - is this covered?",
            "What is the No Claim Discount (NCD) offered in this policy?"
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print("Status Code:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_structured_analysis():
    """Test the enhanced structured analysis endpoint"""
    payload = {
        "documents": """
        POLICY DOCUMENT
        
        Coverage: This policy covers medical expenses for hospitalization, surgery, and emergency treatments.
        
        Waiting Period: 
        - General treatments: No waiting period
        - Pre-existing conditions: 36 months
        - Specific surgeries (knee, hip): 24 months
        
        Age Limits: Coverage available for individuals aged 18-65 years.
        
        Geographic Coverage: Valid across India, including major cities like Mumbai, Delhi, Pune, Bangalore.
        
        Policy Duration Requirements:
        - Minimum policy duration for surgery coverage: 12 months
        - Emergency treatments: Covered from day 1
        
        Exclusions:
        - Cosmetic surgery
        - Experimental treatments
        - Pre-existing conditions during waiting period
        """,
        "questions": [
            "46-year-old male, knee surgery in Pune, 3-month policy",
            "Emergency appendectomy for 25-year-old female, 2-day-old policy",
            "Pre-existing diabetes treatment for 55-year-old, 40-month policy"
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print("Structured Analysis Status:", response.status_code)
        print("Structured Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    
    except Exception as e:
        print(f"Structured analysis test failed: {e}")
        return False

async def async_test_multiple_queries():
    """Test multiple queries asynchronously"""
    test_cases = [
        {
            "documents": "Policy covers dental treatment with 6-month waiting period. Age limit: 18-70 years.",
            "questions": ["Is dental treatment covered for a 25-year-old with 8-month policy?"]
        },
        {
            "documents": "Maternity coverage requires 24-month continuous policy. Maximum coverage: Rs. 50,000.",
            "questions": ["Maternity expenses for 28-year-old with 30-month policy?"]
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for case in test_cases:
            task = session.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=case
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            print(f"\nAsync Test Case {i+1}:")
            print("Status:", response.status)
            result = await response.json()
            print("Response:", json.dumps(result, indent=2))

def test_edge_cases():
    """Test edge cases and error handling"""
    edge_cases = [
        {
            "name": "Empty document",
            "payload": {"documents": "", "questions": ["Is surgery covered?"]}
        },
        {
            "name": "Invalid URL",
            "payload": {"documents": "https://invalid-url.com/doc.pdf", "questions": ["Test question"]}
        },
        {
            "name": "Very long query",
            "payload": {
                "documents": "Short policy text",
                "questions": ["This is a very long question that contains many details about a complex medical procedure involving multiple specialists and requiring extensive hospitalization over several months with various complications and additional treatments that may or may not be covered under the current policy terms and conditions"]
            }
        }
    ]
    
    for case in edge_cases:
        print(f"\nTesting: {case['name']}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=case['payload'],
                timeout=30
            )
            print("Status:", response.status_code)
            if response.status_code != 200:
                print("Error Response:", response.text)
        except Exception as e:
            print(f"Expected error: {e}")

def run_comprehensive_tests():
    """Run all tests"""
    print("🚀 Starting Comprehensive API Tests")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    health_ok = test_health_check()
    print("✅ Health Check Passed" if health_ok else "❌ Health Check Failed")
    
    # Test 2: Main endpoint
    print("\n2. Testing Main HackRX Endpoint...")
    hackrx_ok = test_hackrx_endpoint()
    print("✅ HackRX Endpoint Passed" if hackrx_ok else "❌ HackRX Endpoint Failed")
    
    # Test 3: Structured analysis
    print("\n3. Testing Structured Analysis...")
    structured_ok = test_structured_analysis()
    print("✅ Structured Analysis Passed" if structured_ok else "❌ Structured Analysis Failed")
    
    # Test 4: Edge cases
    print("\n4. Testing Edge Cases...")
    test_edge_cases()
    
    # Test 5: Async tests
    print("\n5. Running Async Tests...")
    try:
        asyncio.run(async_test_multiple_queries())
        print("✅ Async Tests Completed")
    except Exception as e:
        print(f"❌ Async Tests Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test Suite Completed")

def curl_examples():
    """Print curl command examples"""
    print("\n📝 CURL Command Examples:")
    print("=" * 40)
    
    print("\n1. Health Check:")
    print(f"curl -X GET {API_BASE_URL.replace('/api/v1', '')}/health")
    
    print("\n2. Main Query:")
    print(f"""curl -X POST {API_BASE_URL}/hackrx/run \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {AUTH_TOKEN}" \\
  -d '{{
    "documents": "Policy covers surgery with 12-month waiting period. Age: 18-65.",
    "questions": ["Is knee surgery covered for 46-year-old with 3-month policy?"]
  }}'""")
    
    print("\n3. Structured Analysis:")
    print(f"""curl -X POST {API_BASE_URL}/analyze \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {AUTH_TOKEN}" \\
  -d '{{
    "documents": "Insurance policy text here...",
    "questions": ["Your question here"]
  }}'""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Document Analysis API")
    parser.add_argument("--url", default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument("--curl-only", action="store_true", help="Show curl examples only")
    parser.add_argument("--health-only", action="store_true", help="Run health check only")
    
    args = parser.parse_args()
    
    # Update API URL
    API_BASE_URL = args.url
    
    if args.curl_only:
        curl_examples()
    elif args.health_only:
        test_health_check()
    else:
        run_comprehensive_tests()
        curl_examples()