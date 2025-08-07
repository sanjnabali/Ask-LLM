# 🚀 Document Analysis System

A production-ready LLM-powered system for processing natural language queries on unstructured documents such as insurance policies, legal contracts, and compliance documents.

## ✨ Features

- **Multi-format Document Processing**: PDF, DOCX, HTML, and plain text
- **Semantic Search**: Vector embeddings with Pinecone for accurate clause retrieval
- **Natural Language Query Processing**: Parse complex queries like "46M, knee surgery, Pune, 3-month policy"
- **LLM-powered Analysis**: Gemini AI for intelligent decision making and reasoning
- **Structured JSON Responses**: Clean, parseable outputs with decision rationale
- **Real-time Processing**: Sub-3 second response times for most queries
- **Explainable AI**: Trace every decision back to specific document clauses

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client    │───▶│   FastAPI    │───▶│  Doc Processor  │
│  (Query)    │    │   Gateway    │    │  (PDF/DOCX)     │
└─────────────┘    └──────────────┘    └─────────────────┘
                           │                       │
                           ▼                       ▼
                   ┌──────────────┐    ┌─────────────────┐
                   │ LLM Parser   │    │  Text Chunker   │
                   │  (Gemini)    │    │  (Semantic)     │
                   └──────────────┘    └─────────────────┘
                           │                       │
                           ▼                       ▼
                   ┌──────────────┐    ┌─────────────────┐
                   │    Vector    │    │   Embeddings    │
                   │  Search DB   │    │  (Sentence-T)   │
                   │(Pinecone)    │    └─────────────────┘
                   └──────────────┘             │
                           │                    │
                           ▼                    ▼
                   ┌──────────────┐    ┌─────────────────┐
                   │ Decision     │    │  Clause         │
                   │ Engine       │    │  Matching       │
                   └──────────────┘    └─────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ JSON Output  │
                   │  Response    │
                   └──────────────┘
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd document-analysis-system
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys:
# GEMINI_API_KEY=your_gemini_key
# PINECONE_API_KEY=your_pinecone_key
```

### 3. Run Locally

```bash
python main.py
# Server starts on http://localhost:8000
```

### 4. Test the API

```bash
curl -X GET http://localhost:8000/health
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Main Processing Endpoint
```http
POST /api/v1/hackrx/run
Authorization: Bearer 4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc
Content-Type: application/json
```

**Request Body:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the waiting period for pre-existing diseases?",
    "46M, knee surgery, Pune, 3-month policy - covered?",
    "Maternity coverage conditions and limits?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The waiting period for pre-existing diseases is 36 months of continuous coverage from policy inception.",
    "Knee surgery is not covered as the policy duration of 3 months does not meet the minimum 24-month requirement for specific surgeries.",
    "Maternity expenses are covered after 24 months of continuous coverage, limited to two deliveries per policy period."
  ]
}
```

### Enhanced Analysis Endpoint
```http
POST /api/v1/analyze
```

**Response includes structured decision data:**
```json
{
  "results": [
    {
      "question": "46M, knee surgery, Pune, 3-month policy",
      "structured_query": {
        "age": 46,
        "gender": "male",
        "procedure": "knee surgery",
        "location": "Pune",
        "policy_duration": "3 months"
      },
      "decision": "not_covered",
      "analysis": "Coverage denied due to insufficient policy duration...",
      "relevant_clauses": [
        {
          "text": "Specific surgeries require 24 months continuous coverage...",
          "score": 0.89,
          "chunk_id": "policy_chunk_15"
        }
      ],
      "confidence": 0.91
    }
  ]
}
```

## 🔧 Technical Components

### Document Processing
- **PDF**: PyMuPDF for robust text extraction
- **DOCX**: python-docx for Word document parsing  
- **HTML/Email**: BeautifulSoup for web content
- **URL Handling**: Secure HTTPS download with SSL verification

### Semantic Chunking
- **Algorithm**: Sentence-boundary aware chunking
- **Size**: 400 words max with 50-word overlap
- **Optimization**: Preserves context while enabling precise retrieval

### Vector Search
- **Primary**: Pinecone serverless vector database
- **Fallback**: In-memory FAISS for development
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Similarity**: Cosine similarity with semantic reranking

### LLM Integration
- **Model**: Gemini 1.5 Flash for fast, accurate responses
- **Safety**: Configured harm filters for production use
- **Optimization**: Structured prompts for consistent output
- **Fallback**: Graceful degradation when LLM unavailable

## 🎯 Use Cases

### Insurance Claims Processing
```bash
curl -X POST https://your-api.com/api/v1/hackrx/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://company.com/health-policy.pdf",
    "questions": [
      "52F, cardiac surgery in Mumbai, 18-month policy",
      "Pre-existing hypertension coverage after 3 years",
      "Emergency room visit coverage limits"
    ]
  }'
```

### Legal Contract Analysis
```bash
curl -X POST https://your-api.com/api/v1/hackrx/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "Employment contract text here...",
    "questions": [
      "What is the notice period for termination?",
      "Are remote work arrangements permitted?",
      "Non-compete clause duration and scope?"
    ]
  }'
```

### HR Policy Queries
```bash
curl -X POST https://your-api.com/api/v1/hackrx/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "Company handbook content...",
    "questions": [
      "Parental leave policy for new fathers",
      "Training budget allocation per employee",
      "Performance review cycle and criteria"
    ]
  }'
```

## 📊 Performance Metrics


## 🚢 Deployment

### Render (Recommended)
1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy automatically with `render.yaml` configuration




### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional
PINECONE_ENVIRONMENT=gcp-starter
DATABASE_URL=postgresql://user:pass@host:5432/db
PORT=8000
LOG_LEVEL=INFO
```

## 🧪 Testing

### Run All Tests
```bash
python test_client.py --url https://your-deployed-api.com/api/v1
```

### Health Check Only
```bash
python test_client.py --health-only --url https://your-api.com/api/v1
```

### Show cURL Examples
```bash
python test_client.py --curl-only
```

### Test Results Example
```
🚀 Starting Comprehensive API Tests
==================================================

1. Testing Health Check...
✅ Health Check Passed

2. Testing Main HackRX Endpoint...
Status Code: 200
✅ HackRX Endpoint Passed

3. Testing Structured Analysis...
✅ Structured Analysis Passed

4. Testing Edge Cases...
✅ All Edge Cases Handled

5. Running Async Tests...
✅ Async Tests Completed

🏁 Test Suite Completed
```

## 🔍 Query Examples

### Simple Eligibility Check
**Input:** "Is dental treatment covered?"
**Output:** "Yes, dental treatment is covered under the policy with a 6-month waiting period for new policies."

### Complex Multi-factor Query  
**Input:** "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
**Output:** "Coverage denied. Knee surgery requires a minimum 24-month continuous policy period. Current policy duration of 3 months is insufficient."

### Procedure-specific Query
**Input:** "Cataract surgery waiting period and coverage limits"
**Output:** "Cataract surgery has a 2-year waiting period. Coverage includes surgeon fees, hospital charges, and lens costs up to policy limits."

## 📈 Optimization Features

### Token Efficiency
- **Smart Chunking**: Only relevant sections processed
- **Query Optimization**: Structured parsing reduces LLM calls
- **Caching**: Vector embeddings cached for repeated queries
- **Early Exit**: High-confidence answers skip additional processing

### Accuracy Boosting
- **Semantic Filtering**: Removes irrelevant document noise  
- **Multi-stage Retrieval**: Embedding + keyword + semantic reranking
- **Context Preservation**: Overlap maintains clause relationships
- **Confidence Scoring**: Reliability metrics for each answer

### Latency Optimization
- **Async Processing**: Concurrent document processing
- **Vector Caching**: Pre-computed embeddings for common documents
- **Connection Pooling**: Efficient database connections
- **Response Streaming**: Progressive result delivery

## 🛡️ Security & Compliance

### Authentication
- Bearer token authentication required
- API key validation on all endpoints
- Request rate limiting (configurable)

### Data Privacy  
- No persistent storage of sensitive documents
- In-memory processing with automatic cleanup
- GDPR compliant data handling
- Audit logging for compliance tracking

### Error Handling
- Graceful degradation when services unavailable
- Comprehensive error messages with request IDs
- Automatic retry logic for transient failures
- Health monitoring and alerting

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure all tests pass: `python -m pytest`
5. Submit pull request with detailed description

## 📄 License

MIT License - See LICENSE file for details

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Documentation**: API documentation at `/docs` endpoint
- **Email**: support@document-analysis.com

---

