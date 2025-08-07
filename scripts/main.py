# main.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import hashlib
import re

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Document processing
import PyMuPDF  # fitz
from docx import Document
from bs4 import BeautifulSoup
import requests

# Vector DB and embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sentence_transformers import SentenceTransformer

# Database
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncpg

# LLM
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Utils
from urllib.parse import urlparse
import aiohttp
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
DATABASE_URL = os.getenv("DATABASE_URL")
AUTH_TOKEN = "4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc"

# Initialize services
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Document Analysis System",
    description="LLM-powered system for processing natural language queries on unstructured documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="Document URL or text content")
    questions: List[str] = Field(..., description="List of questions to answer")

class StructuredQuery(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    raw_query: str

class ClauseMatch(BaseModel):
    text: str
    score: float
    source: str
    chunk_id: str

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: str
    clauses_used: List[ClauseMatch]
    confidence: float

# Document Processing Classes
class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.html', '.txt']
    
    async def process_document(self, source: str) -> str:
        """Process document from URL or direct text"""
        if self._is_url(source):
            return await self._process_from_url(source)
        else:
            return source  # Direct text content
    
    def _is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def _process_from_url(self, url: str) -> str:
        """Download and process document from URL"""
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            ) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download document: {response.status}")
                    
                    content = await response.read()
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'pdf' in content_type:
                        return self._extract_pdf_text(content)
                    elif 'word' in content_type or 'docx' in content_type:
                        return self._extract_docx_text(content)
                    elif 'html' in content_type:
                        return self._extract_html_text(content.decode('utf-8'))
                    else:
                        return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error processing document from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            doc = PyMuPDF.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(status_code=400, detail="Error processing PDF")
    
    def _extract_docx_text(self, docx_content: bytes) -> str:
        """Extract text from DOCX bytes"""
        try:
            import io
            doc = Document(io.BytesIO(docx_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise HTTPException(status_code=400, detail="Error processing DOCX")
    
    def _extract_html_text(self, html_content: str) -> str:
        """Extract text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text()
        except Exception as e:
            logger.error(f"Error extracting HTML text: {e}")
            return html_content

class TextChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source: str = "document") -> List[Dict[str, Any]]:
        """Split text into semantic chunks with overlap"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    "id": f"{source}_chunk_{chunk_id}",
                    "text": current_chunk.strip(),
                    "source": source,
                    "word_count": len(current_chunk.split())
                })
                
                # Add overlap
                overlap_sentences = current_chunk.split('. ')[-2:]
                current_chunk = '. '.join(overlap_sentences) + ". " if len(overlap_sentences) > 1 else ""
                chunk_id += 1
            
            current_chunk += sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "id": f"{source}_chunk_{chunk_id}",
                "text": current_chunk.strip(),
                "source": source,
                "word_count": len(current_chunk.split())
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

class VectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone and embedding model"""
        try:
            if PINECONE_API_KEY:
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                self.index_name = "document-analysis"
                
                # Create index if it doesn't exist
                existing_indexes = self.pc.list_indexes().names()
                if self.index_name not in existing_indexes:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                
                self.index = self.pc.Index(self.index_name)
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Fallback to in-memory search if Pinecone fails
            self.chunks_store = []
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed and store chunks in vector database"""
        try:
            if self.index:
                vectors_to_upsert = []
                for chunk in chunks:
                    embedding = self.embedding_model.encode(chunk["text"]).tolist()
                    vectors_to_upsert.append({
                        "id": chunk["id"],
                        "values": embedding,
                        "metadata": {
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "word_count": chunk["word_count"]
                        }
                    })
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    self.index.upsert(vectors=batch)
            else:
                # Fallback to in-memory storage
                self.chunks_store = chunks
                for chunk in chunks:
                    chunk["embedding"] = self.embedding_model.encode(chunk["text"])
        
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            # Fallback to in-memory storage
            self.chunks_store = chunks
            for chunk in chunks:
                chunk["embedding"] = self.embedding_model.encode(chunk["text"])
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[ClauseMatch]:
        """Search for similar chunks using vector similarity"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            if self.index:
                # Use Pinecone
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                matches = []
                for match in results.matches:
                    matches.append(ClauseMatch(
                        text=match.metadata["text"],
                        score=float(match.score),
                        source=match.metadata["source"],
                        chunk_id=match.id
                    ))
                return matches
            
            else:
                # Fallback to in-memory cosine similarity
                if not hasattr(self, 'chunks_store'):
                    return []
                
                query_emb = np.array(query_embedding)
                similarities = []
                
                for chunk in self.chunks_store:
                    chunk_emb = np.array(chunk["embedding"])
                    similarity = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
                    similarities.append((similarity, chunk))
                
                # Sort by similarity and get top_k
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                matches = []
                for score, chunk in similarities[:top_k]:
                    matches.append(ClauseMatch(
                        text=chunk["text"],
                        score=float(score),
                        source=chunk["source"],
                        chunk_id=chunk["id"]
                    ))
                return matches
        
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []

class LLMProcessor:
    def __init__(self):
        if GEMINI_API_KEY:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
    
    async def parse_query(self, query: str) -> StructuredQuery:
        """Parse natural language query into structured format"""
        if not self.model:
            # Fallback parsing
            return StructuredQuery(raw_query=query)
        
        prompt = f"""
        Parse this natural language query and extract structured information:
        Query: "{query}"
        
        Extract the following fields if present:
        - age: integer
        - gender: male/female
        - procedure: medical procedure or service
        - location: city/state/country
        - policy_duration: time period
        
        Return ONLY a JSON object with these fields. Use null for missing fields.
        Example: {{"age": 46, "gender": "male", "procedure": "knee surgery", "location": "Pune", "policy_duration": "3 months", "raw_query": "{query}"}}
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            response_text = response.text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return StructuredQuery(**parsed_data)
            else:
                return StructuredQuery(raw_query=query)
        
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return StructuredQuery(raw_query=query)
    
    async def analyze_and_decide(self, query: StructuredQuery, relevant_chunks: List[ClauseMatch]) -> str:
        """Analyze retrieved chunks and make a decision"""
        if not self.model:
            return "Analysis not available - LLM not configured"
        
        chunks_text = "\n\n".join([f"Clause {i+1}: {chunk.text}" for i, chunk in enumerate(relevant_chunks)])
        
        prompt = f"""
        You are an expert document analyst for insurance, legal, and compliance domains.
        
        QUERY DETAILS:
        - Raw Query: {query.raw_query}
        - Age: {query.age}
        - Gender: {query.gender}
        - Procedure: {query.procedure}
        - Location: {query.location}
        - Policy Duration: {query.policy_duration}
        
        RELEVANT CLAUSES:
        {chunks_text}
        
        TASK:
        Analyze the clauses and answer the query. Consider:
        1. Coverage eligibility based on the procedure
        2. Policy duration requirements
        3. Age and location restrictions
        4. Waiting periods
        5. Exclusions and conditions
        
        Provide a clear, direct answer explaining whether the request is covered, any amounts, and the reasoning based on the specific clauses.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return f"Error in analysis: {str(e)}"

# Initialize components
doc_processor = DocumentProcessor()
text_chunker = TextChunker()
vector_store = VectorStore()
llm_processor = LLMProcessor()

# Dependency for authentication
async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(" ")[1]
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/v1/hackrx/run")
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing queries"""
    try:
        # Process document
        document_text = await doc_processor.process_document(request.documents)
        
        # Chunk the document
        chunks = text_chunker.chunk_text(document_text, "policy_document")
        
        # Embed chunks in vector store
        vector_store.embed_chunks(chunks)
        
        # Process each question
        answers = []
        
        for question in request.questions:
            try:
                # Parse the query
                structured_query = await llm_processor.parse_query(question)
                
                # Search for relevant chunks
                relevant_chunks = vector_store.search_similar_chunks(question, top_k=5)
                
                # Analyze and get decision
                answer = await llm_processor.analyze_and_decide(structured_query, relevant_chunks)
                
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        return {"answers": answers}
    
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/v1/analyze")
async def analyze_structured(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Enhanced endpoint returning structured analysis"""
    try:
        # Process document
        document_text = await doc_processor.process_document(request.documents)
        
        # Chunk the document
        chunks = text_chunker.chunk_text(document_text, "policy_document")
        
        # Embed chunks in vector store
        vector_store.embed_chunks(chunks)
        
        # Process each question
        results = []
        
        for question in request.questions:
            try:
                # Parse the query
                structured_query = await llm_processor.parse_query(question)
                
                # Search for relevant chunks
                relevant_chunks = vector_store.search_similar_chunks(question, top_k=5)
                
                # Analyze and get decision
                analysis = await llm_processor.analyze_and_decide(structured_query, relevant_chunks)
                
                # Create structured response
                decision_response = {
                    "question": question,
                    "structured_query": structured_query.dict(),
                    "decision": "covered" if any(word in analysis.lower() for word in ["yes", "covered", "eligible"]) else "not_covered",
                    "analysis": analysis,
                    "relevant_clauses": [
                        {
                            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            "score": chunk.score,
                            "chunk_id": chunk.chunk_id
                        }
                        for chunk in relevant_chunks
                    ],
                    "confidence": min(relevant_chunks[0].score if relevant_chunks else 0.5, 1.0)
                }
                
                results.append(decision_response)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                results.append({
                    "question": question,
                    "error": f"Processing error: {str(e)}"
                })
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error in structured analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )