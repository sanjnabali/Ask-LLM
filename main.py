# main.py - Python 3.12.4 compatible version
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio
import hashlib
import re
import time
import traceback
import sys

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

# Document processing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF not available, PDF processing disabled")

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
    print("Warning: python-docx not available, DOCX processing disabled")

from bs4 import BeautifulSoup
import requests

# Vector DB and embeddings - Updated imports for Python 3.12
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: Pinecone not available, using fallback vector search")

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: Sentence Transformers not available")

# LLM - Updated for Python 3.12
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available")

# Utils
from urllib.parse import urlparse
import aiohttp
import ssl
import certifi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Environment variables with fallbacks
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
DATABASE_URL = os.getenv("DATABASE_URL")
PORT = int(os.getenv("PORT", 8000))
AUTH_TOKEN = os.getenv("AUTH_TOKEN")


# Initialize Gemini if available
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini AI: {e}")
        GEMINI_AVAILABLE = False

logger.info(f"Starting server setup")
logger.info(f"Python version: {sys.version}")
logger.info(f"Gemini available: {GEMINI_API_KEY is not None and GEMINI_AVAILABLE}")
logger.info(f"Pinecone available: {PINECONE_API_KEY is not None and PINECONE_AVAILABLE}")
logger.info(f"Sentence Transformers available: {SENTENCE_TRANSFORMERS_AVAILABLE}")

# FastAPI app
app = FastAPI(
    title="Ask-LLM",
    description="LLM-powered system for processing natural language queries on unstructured documents",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated Pydantic models for v2 compatibility
class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    documents: str = Field(..., description="Document URL or text content", min_length=1)
    questions: List[str] = Field(..., description="List of questions to answer", min_length=1)

class StructuredQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
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

# Document Processing Classes
class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.html', '.txt']
        self.session_timeout = 30
    
    async def process_document(self, source: str) -> str:
        """Process document from URL or direct text"""
        try:
            if self._is_url(source):
                return await self._process_from_url(source)
            else:
                return source  # Direct text content
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
    
    def _is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def _process_from_url(self, url: str) -> str:
        """Download and process document from URL"""
        try:
            # Create SSL context with better configuration for Python 3.12
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE  # For development - enable in production
            
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit=100, limit_per_host=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; DocumentAnalysisBot/1.0)'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to download document: HTTP {response.status}"
                        )
                    
                    content = await response.read()
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'pdf' in content_type:
                        return self._extract_pdf_text(content)
                    elif 'word' in content_type or 'docx' in content_type:
                        return self._extract_docx_text(content)
                    elif 'html' in content_type:
                        return self._extract_html_text(content.decode('utf-8', errors='ignore'))
                    else:
                        return content.decode('utf-8', errors='ignore')
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error processing document from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing document from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        if not fitz:
            raise HTTPException(status_code=400, detail="PDF processing not available")
        
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(status_code=400, detail="Error processing PDF")
    
    def _extract_docx_text(self, docx_content: bytes) -> str:
        """Extract text from DOCX bytes"""
        if not DocxDocument:
            raise HTTPException(status_code=400, detail="DOCX processing not available")
        
        try:
            import io
            doc = DocxDocument(io.BytesIO(docx_content))
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
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.error(f"Error extracting HTML text: {e}")
            return html_content

class TextChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source: str = "document") -> List[Dict[str, Any]]:
        """Split text into semantic chunks with overlap"""
        try:
            sentences = self._split_into_sentences(text)
            chunks = []
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "id": f"{source}_chunk_{chunk_id}",
                        "text": current_chunk.strip(),
                        "source": source,
                        "word_count": len(current_chunk.split())
                    })
                    
                    # Create overlap for next chunk
                    words = current_chunk.split()
                    overlap_words = words[-self.overlap:] if len(words) > self.overlap else words
                    current_chunk = " ".join(overlap_words)
                    chunk_id += 1
                
                current_chunk = potential_chunk
            
            # Add final chunk if it exists
            if current_chunk.strip():
                chunks.append({
                    "id": f"{source}_chunk_{chunk_id}",
                    "text": current_chunk.strip(),
                    "source": source,
                    "word_count": len(current_chunk.split())
                })
            
            return chunks if chunks else [{
                "id": f"{source}_chunk_0",
                "text": text[:self.chunk_size],
                "source": source,
                "word_count": len(text.split())
            }]
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Return single chunk as fallback
            return [{
                "id": f"{source}_chunk_0",
                "text": text[:self.chunk_size],
                "source": source,
                "word_count": len(text.split())
            }]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex - improved for Python 3.12"""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences if cleaned_sentences else [text]

class VectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self.embedding_model = None
        self.chunks_store = []  # Fallback storage
        self.embedding_dim = 384  # for all-MiniLM-L6-v2
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone and embedding model"""
        # Initialize embedding model first
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            global SENTENCE_TRANSFORMERS_AVAILABLE
            SENTENCE_TRANSFORMERS_AVAILABLE = False  # Correctly modifies global variable

        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence Transformers not available - using dummy embeddings")
            self.embedding_model = self._create_dummy_embedding_model()
        
        # Try to initialize Pinecone
        if PINECONE_AVAILABLE and PINECONE_API_KEY:
            try:
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                self.index_name = "askllm"
                
                # Check if index exists
                try:
                    existing_indexes = self.pc.list_indexes()
                    index_names = [idx.name for idx in existing_indexes]
                    
                    if self.index_name not in index_names:
                        logger.info(f"Creating new Pinecone index: {self.index_name}")
                        self.pc.create_index(
                            name=self.index_name,
                            dimension=self.embedding_dim,
                            metric="cosine",
                            spec=ServerlessSpec(
                                cloud="aws",
                                region="us-east-1"
                            )
                        )
                        # Wait for index to be ready
                        time.sleep(10)
                    
                    self.index = self.pc.Index(self.index_name)
                    logger.info("Pinecone initialized successfully")
                    
                except Exception as e:
                    logger.warning(f"Pinecone index operation failed: {e}")
                    self.pc = None
                    self.index = None
                    
            except Exception as e:
                logger.warning(f"Pinecone initialization failed, using fallback: {e}")
                self.pc = None
                self.index = None
        else:
            logger.info("Using in-memory vector search (Pinecone not available)")
    
    def _create_dummy_embedding_model(self):
        """Create a dummy embedding model for fallback"""
        class DummyEmbeddingModel:
            def encode(self, text, **kwargs):
                if isinstance(text, str):
                    # Simple hash-based embedding
                    hash_val = hashlib.md5(text.encode()).hexdigest()
                    return np.array([float(int(hash_val[i:i+2], 16)) / 255.0 
                                   for i in range(0, min(len(hash_val), self.embedding_dim*2), 2)]
                                  + [0.0] * (self.embedding_dim - min(len(hash_val)//2, self.embedding_dim)))
                elif isinstance(text, list):
                    return [self.encode(t) for t in text]
                else:
                    return np.zeros(self.embedding_dim)
        
        return DummyEmbeddingModel()
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed and store chunks in vector database"""
        try:
            if self.index and PINECONE_AVAILABLE:
                vectors_to_upsert = []
                for chunk in chunks:
                    try:
                        embedding = self.embedding_model.encode(chunk["text"])
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        elif isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        
                        vectors_to_upsert.append({
                            "id": chunk["id"],
                            "values": embedding,
                            "metadata": {
                                "text": chunk["text"][:1000],  # Limit metadata size
                                "source": chunk["source"],
                                "word_count": chunk["word_count"]
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to embed chunk {chunk['id']}: {e}")
                        continue
                
                if vectors_to_upsert:
                    # Upsert in batches
                    batch_size = 100
                    for i in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[i:i+batch_size]
                        try:
                            self.index.upsert(vectors=batch)
                        except Exception as e:
                            logger.error(f"Failed to upsert batch {i//batch_size}: {e}")
                
                logger.info(f"Successfully embedded {len(vectors_to_upsert)} chunks to Pinecone")
            else:
                # Fallback to in-memory storage
                self.chunks_store = chunks.copy()
                for chunk in self.chunks_store:
                    try:
                        embedding = self.embedding_model.encode(chunk["text"])
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        elif isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        chunk["embedding"] = embedding
                    except Exception as embed_error:
                        logger.error(f"Failed to embed chunk: {embed_error}")
                        # Use zero vector as fallback
                        chunk["embedding"] = [0.0] * self.embedding_dim
                        
                logger.info(f"Stored {len(chunks)} chunks in memory")
        
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            # Always fallback to in-memory storage
            self.chunks_store = chunks.copy()
            for chunk in self.chunks_store:
                try:
                    embedding = self.embedding_model.encode(chunk["text"])
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    chunk["embedding"] = embedding
                except Exception as embed_error:
                    logger.error(f"Failed to embed chunk: {embed_error}")
                    chunk["embedding"] = [0.0] * self.embedding_dim
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[ClauseMatch]:
        """Search for similar chunks using vector similarity"""
        try:
            query_embedding = self.embedding_model.encode(query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            elif isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            if self.index and PINECONE_AVAILABLE:
                try:
                    # Use Pinecone
                    results = self.index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True
                    )
                    
                    matches = []
                    for match in results.matches:
                        matches.append(ClauseMatch(
                            text=match.metadata.get("text", ""),
                            score=float(match.score),
                            source=match.metadata.get("source", "unknown"),
                            chunk_id=match.id
                        ))
                    return matches
                except Exception as e:
                    logger.warning(f"Pinecone search failed, using fallback: {e}")
            
            # Fallback to in-memory cosine similarity
            if not self.chunks_store:
                return []
            
            query_emb = np.array(query_embedding)
            similarities = []
            
            for chunk in self.chunks_store:
                try:
                    chunk_emb = np.array(chunk["embedding"])
                    # Cosine similarity
                    dot_product = np.dot(query_emb, chunk_emb)
                    norm_query = np.linalg.norm(query_emb)
                    norm_chunk = np.linalg.norm(chunk_emb)
                    
                    if norm_query > 0 and norm_chunk > 0:
                        similarity = dot_product / (norm_query * norm_chunk)
                    else:
                        similarity = 0.0
                    
                    similarities.append((similarity, chunk))
                except Exception as e:
                    logger.warning(f"Error computing similarity: {e}")
                    similarities.append((0.0, chunk))
            
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
        self.model = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
    
    async def parse_query(self, query: str) -> StructuredQuery:
        """Parse natural language query into structured format"""
        if not self.model or not GEMINI_AVAILABLE:
            # Fallback parsing using regex
            return self._fallback_parse_query(query)
        
        prompt = f"""
        Parse this natural language query and extract structured information:
        Query: "{query}"
        
        Extract the following fields if present:
        - age: integer (extract number followed by 'year', 'M', 'F', or age indicators)
        - gender: "male" or "female" (look for M, F, male, female indicators)
        - procedure: medical procedure or service name
        - location: city, state, or country name
        - policy_duration: time period (like "3 months", "2 years", etc.)
        
        Return ONLY a valid JSON object with these fields. Use null for missing fields.
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
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed_data = json.loads(json_match.group())
                    parsed_data["raw_query"] = query  # Ensure raw_query is set
                    return StructuredQuery(**parsed_data)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {e}")
                    return self._fallback_parse_query(query)
            else:
                return self._fallback_parse_query(query)
        
        except Exception as e:
            logger.error(f"Error parsing query with LLM: {e}")
            return self._fallback_parse_query(query)
    
    def _fallback_parse_query(self, query: str) -> StructuredQuery:
        """Fallback query parsing using regex"""
        structured_data = {"raw_query": query}
        
        # Extract age
        age_match = re.search(r'(\d+)[-\s]*(?:year|yr|M|F|male|female)', query, re.IGNORECASE)
        if age_match:
            try:
                structured_data["age"] = int(age_match.group(1))
            except ValueError:
                pass
        
        # Extract gender
        if re.search(r'\b(male|M)\b', query, re.IGNORECASE) and not re.search(r'\b(female|F)\b', query, re.IGNORECASE):
            structured_data["gender"] = "male"
        elif re.search(r'\b(female|F)\b', query, re.IGNORECASE):
            structured_data["gender"] = "female"
        
        # Extract procedures (common medical terms)
        procedure_patterns = [
            r'(\w+\s+surgery)', r'(\w+\s+treatment)', r'(\w+\s+procedure)',
            r'(\w+\s+operation)', r'(\w+\s+therapy)', r'(surgery)', r'(treatment)'
        ]
        for pattern in procedure_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                structured_data["procedure"] = match.group(1).lower()
                break
        
        # Extract location (common Indian cities)
        cities = ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata', 'hyderabad']
        for city in cities:
            if city in query.lower():
                structured_data["location"] = city.title()
                break
        
        # Extract policy duration
        duration_match = re.search(r'(\d+)[-\s]*(month|year|day)s?', query, re.IGNORECASE)
        if duration_match:
            structured_data["policy_duration"] = f"{duration_match.group(1)} {duration_match.group(2).lower()}s"
        
        return StructuredQuery(**structured_data)
    
    async def analyze_and_decide(self, query: StructuredQuery, relevant_chunks: List[ClauseMatch]) -> str:
        """Analyze retrieved chunks and make a decision"""
        if not self.model or not GEMINI_AVAILABLE:
            return self._fallback_analyze(query, relevant_chunks)
        
        chunks_text = "\n\n".join([
            f"Clause {i+1} (Score: {chunk.score:.2f}): {chunk.text[:500]}" 
            for i, chunk in enumerate(relevant_chunks[:3])  # Limit to top 3 chunks
        ])
        
        prompt = f"""
        You are an expert document analyst for insurance, legal, and compliance domains.
        
        QUERY DETAILS:
        - Raw Query: {query.raw_query}
        - Age: {query.age or "Not specified"}
        - Gender: {query.gender or "Not specified"}
        - Procedure: {query.procedure or "Not specified"}
        - Location: {query.location or "Not specified"}
        - Policy Duration: {query.policy_duration or "Not specified"}
        
        RELEVANT DOCUMENT CLAUSES:
        {chunks_text}
        
        TASK:
        Analyze the clauses and answer the query directly and concisely. Consider:
        1. Coverage eligibility based on the procedure
        2. Policy duration requirements
        3. Age and location restrictions
        4. Waiting periods
        5. Exclusions and conditions
        
        Provide a clear, direct answer (2-3 sentences max) explaining whether the request is covered and the key reasoning.
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
            return self._fallback_analyze(query, relevant_chunks)
    
    def _fallback_analyze(self, query: StructuredQuery, relevant_chunks: List[ClauseMatch]) -> str:
        """Fallback analysis without LLM"""
        if not relevant_chunks:
            return "No relevant information found in the document to answer the query."
        
        # Simple rule-based analysis
        top_chunk = relevant_chunks[0]
        query_text = query.raw_query.lower()
        chunk_text = top_chunk.text.lower()
        
        # Basic coverage check
        if any(word in query_text for word in ['covered', 'coverage', 'eligible']):
            if any(word in chunk_text for word in ['covered', 'eligible', 'included']):
                return f"Based on the policy, this appears to be covered. Relevant clause: {top_chunk.text[:200]}..."
            elif any(word in chunk_text for word in ['excluded', 'not covered', 'waiting']):
                return f"This may not be covered or may have restrictions. Relevant clause: {top_chunk.text[:200]}..."
        
        # Default response
        return f"Based on the policy documents: {top_chunk.text[:300]}..."

# Initialize components
doc_processor = DocumentProcessor()
text_chunker = TextChunker()
vector_store = VectorStore()
llm_processor = LLMProcessor()

# Dependency for authentication
async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(" ")[1]
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    return {
        "status": "ok", 
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "services": {
            "gemini": "available" if GEMINI_AVAILABLE and GEMINI_API_KEY else "not_configured",
            "pinecone": "available" if PINECONE_AVAILABLE and PINECONE_API_KEY else "fallback_mode",
            "sentence_transformers": "available" if SENTENCE_TRANSFORMERS_AVAILABLE else "fallback_mode",
            "embedding_model": "loaded" if vector_store.embedding_model else "error"
        }
    }

@app.post("/api/v1/hackrx/run")
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing queries"""
    start_time = time.time()
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Process document
        document_text = await doc_processor.process_document(request.documents)
        logger.info(f"Document processed, length: {len(document_text)} characters")
        
        # Chunk the document
        chunks = text_chunker.chunk_text(document_text, "policy_document")
        logger.info(f"Document chunked into {len(chunks)} pieces")
        
        # Embed chunks in vector store
        vector_store.embed_chunks(chunks)
        logger.info("Chunks embedded successfully")
        
        # Process each question
        answers = []
        
        for i, question in enumerate(request.questions):
            try:
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                
                # Parse the query
                structured_query = await llm_processor.parse_query(question)
                
                # Search for relevant chunks
                relevant_chunks = vector_store.search_similar_chunks(question, top_k=5)
                logger.info(f"Found {len(relevant_chunks)} relevant chunks")
                
                # Analyze and get decision
                answer = await llm_processor.analyze_and_decide(structured_query, relevant_chunks)
                
                answers.append(answer)
                logger.info(f"Question {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                logger.error(traceback.format_exc())
                answers.append(f"Error processing question: {str(e)}")
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return {"answers": answers}
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in main processing after {processing_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
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
                    "structured_query": structured_query.model_dump(),  # Updated for Pydantic v2
                    "decision": "covered" if any(word in analysis.lower() for word in ["yes", "covered", "eligible"]) else "not_covered",
                    "analysis": analysis,
                    "relevant_clauses": [
                        {
                            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            "score": chunk.score,
                            "chunk_id": chunk.chunk_id
                        }
                        for chunk in relevant_chunks[:3]  # Limit to top 3
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

# Error handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error. Please check your request format."}
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

