# optimizations.py - Production-ready optimizations and caching

import asyncio
import hashlib
import pickle
import time
from functools import wraps, lru_cache
from typing import Dict, List, Any, Optional
import redis
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based caching for embeddings and responses"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=False
            )
            self.redis_available = True
        except:
            self.redis_available = False
            self.memory_cache = {}
    
    def _get_cache_key(self, prefix: str, content: str) -> str:
        """Generate cache key from content hash"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{content_hash}"
    
    def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        cache_key = self._get_cache_key("embedding", text)
        
        try:
            if self.redis_available:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            else:
                return self.memory_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def set_embedding_cache(self, text: str, embedding: List[float], ttl: int = 3600):
        """Cache embedding with TTL"""
        cache_key = self._get_cache_key("embedding", text)
        
        try:
            if self.redis_available:
                self.redis_client.setex(cache_key, ttl, pickle.dumps(embedding))
            else:
                self.memory_cache[cache_key] = embedding
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def get_response_cache(self, query: str, doc_hash: str) -> Optional[Dict]:
        """Get cached response for query+document combination"""
        cache_key = self._get_cache_key("response", f"{query}:{doc_hash}")
        
        try:
            if self.redis_available:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            else:
                return self.memory_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Response cache get error: {e}")
        
        return None
    
    def set_response_cache(self, query: str, doc_hash: str, response: Dict, ttl: int = 1800):
        """Cache response with TTL"""
        cache_key = self._get_cache_key("response", f"{query}:{doc_hash}")
        
        try:
            if self.redis_available:
                self.redis_client.setex(cache_key, ttl, pickle.dumps(response))
            else:
                self.memory_cache[cache_key] = response
        except Exception as e:
            logger.warning(f"Response cache set error: {e}")

# Global cache manager
cache_manager = CacheManager()

def cache_embeddings(func):
    """Decorator to cache embedding computations"""
    @wraps(func)
    async def wrapper(self, text: str, *args, **kwargs):
        # Check cache first
        cached_embedding = cache_manager.get_embedding_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Compute embedding
        embedding = await func(self, text, *args, **kwargs)
        
        # Cache result
        cache_manager.set_embedding_cache(text, embedding)
        
        return embedding
    
    return wrapper

class PerformanceOptimizedVectorStore(VectorStore):
    """Enhanced vector store with caching and optimization"""
    
    @cache_embeddings
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        return await asyncio.to_thread(
            self.embedding_model.encode, 
            text
        )
    
    async def batch_embed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Batch embedding with parallel processing"""
        texts = [chunk["text"] for chunk in chunks]
        
        # Check cache for existing embeddings
        embeddings = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = cache_manager.get_embedding_cache(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
        
        # Compute missing embeddings in batch
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            batch_embeddings = await asyncio.to_thread(
                self.embedding_model.encode,
                uncached_texts
            )
            
            # Fill in computed embeddings and cache them
            for idx, embedding in zip(uncached_indices, batch_embeddings):
                embeddings[idx] = embedding.tolist()
                cache_manager.set_embedding_cache(texts[idx], embeddings[idx])
        
        # Store in vector database
        if self.index:
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                vectors_to_upsert.append({
                    "id": chunk["id"],
                    "values": embeddings[i],
                    "metadata": {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "word_count": chunk["word_count"]
                    }
                })
            
            # Batch upsert
            await self._batch_upsert(vectors_to_upsert)
    
    async def _batch_upsert(self, vectors: List[Dict], batch_size: int = 100):
        """Upsert vectors in batches asynchronously"""
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            await asyncio.to_thread(self.index.upsert, vectors=batch)

class EarlyExitProcessor:
    """Implements early exit logic for high-confidence answers"""
    
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
    
    def should_exit_early(self, 
                         query: StructuredQuery, 
                         chunks: List[ClauseMatch]) -> bool:
        """Determine if we have enough information for early exit"""
        if not chunks:
            return False
        
        # Check if top match is highly confident
        if chunks[0].score > self.confidence_threshold:
            # Check if query is simple (single concept)
            simple_indicators = [
                len(query.raw_query.split()) <= 8,
                not any(word in query.raw_query.lower() for word in ['and', 'or', 'also', 'additionally']),
                chunks[0].score - chunks[1].score > 0.1 if len(chunks) > 1 else True
            ]
            
            return sum(simple_indicators) >= 2
        
        return False

class AdaptiveChunker(TextChunker):
    """Smart chunker that adapts size based on document type"""
    
    def __init__(self):
        super().__init__()
        self.document_type_patterns = {
            'policy': {
                'indicators': ['policy', 'coverage', 'premium', 'claim'],
                'chunk_size': 300,
                'overlap': 40
            },
            'contract': {
                'indicators': ['agreement', 'contract', 'party', 'terms'],
                'chunk_size': 400,
                'overlap': 50
            },
            'manual': {
                'indicators': ['procedure', 'step', 'instruction', 'guide'],
                'chunk_size': 250,
                'overlap': 30
            }
        }
    
    def detect_document_type(self, text: str) -> str:
        """Detect document type from content"""
        text_lower = text.lower()
        type_scores = {}
        
        for doc_type, config in self.document_type_patterns.items():
            score = sum(1 for indicator in config['indicators'] 
                       if indicator in text_lower)
            type_scores[doc_type] = score
        
        # Return type with highest score, default to 'policy'
        return max(type_scores, key=type_scores.get) if type_scores else 'policy'
    
    def chunk_text(self, text: str, source: str = "document") -> List[Dict[str, Any]]:
        """Adaptive chunking based on document type"""
        doc_type = self.detect_document_type(text)
        config = self.document_type_patterns[doc_type]
        
        # Temporarily update chunk size and overlap
        original_chunk_size = self.chunk_size
        original_overlap = self.overlap
        
        self.chunk_size = config['chunk_size']
        self.overlap = config['overlap']
        
        chunks = super().chunk_text(text, source)
        
        # Restore original settings
        self.chunk_size = original_chunk_size
        self.overlap = original_overlap
        
        # Add document type metadata
        for chunk in chunks:
            chunk['document_type'] = doc_type
        
        return chunks

class SmartQueryProcessor:
    """Enhanced query processor with context awareness"""
    
    def __init__(self):
        self.query_patterns = {
            'eligibility': ['covered', 'eligible', 'qualify', 'entitled'],
            'waiting_period': ['waiting', 'period', 'time', 'duration'],
            'limits': ['limit', 'maximum', 'cap', 'restriction'],
            'exclusions': ['exclude', 'not covered', 'exception'],
            'procedures': ['surgery', 'treatment', 'procedure', 'operation']
        }
    
    def enhance_query(self, original_query: str, structured: StructuredQuery) -> str:
        """Enhance query with domain-specific context"""
        query_lower = original_query.lower()
        detected_intents = []
        
        # Detect query intent
        for intent, keywords in self.query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Build enhanced query
        enhanced_parts = [original_query]
        
        if 'eligibility' in detected_intents:
            enhanced_parts.append("coverage eligibility requirements")
        
        if 'waiting_period' in detected_intents:
            enhanced_parts.append("waiting period duration time requirements")
        
        if structured.procedure:
            enhanced_parts.append(f"{structured.procedure} specific requirements")
        
        if structured.age:
            enhanced_parts.append(f"age {structured.age} requirements")
        
        return " ".join(enhanced_parts)

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"Function {func.__name__} executed in {execution_time:.2f}s")
            
            # Alert on slow operations
            if execution_time > 5.0:
                logger.warning(f"Slow operation detected: {func.__name__} took {execution_time:.2f}s")
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper

class RateLimiter:
    """Token bucket rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit"""
        async with self.lock:
            current_time = time.time()
            
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # Clean old requests
            cutoff_time = current_time - self.window_seconds
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff_time
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(current_time)
                return True
            
            return False

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                
                # Reset on success
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                
                raise

# Enhanced main application with optimizations
class OptimizedDocumentAnalysisSystem:
    """Production-optimized document analysis system"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.text_chunker = AdaptiveChunker()
        self.vector_store = PerformanceOptimizedVectorStore()
        self.llm_processor = LLMProcessor()
        self.query_processor = SmartQueryProcessor()
        self.early_exit = EarlyExitProcessor()
        self.rate_limiter = RateLimiter()
        self.circuit_breaker = CircuitBreaker()
    
    @monitor_performance
    async def process_document_optimized(self, document_source: str) -> str:
        """Optimized document processing with caching"""
        # Generate document hash for caching
        doc_hash = hashlib.md5(document_source.encode()).hexdigest()
        
        # Check if document is already processed
        cached_chunks = cache_manager.get_response_cache("document_chunks", doc_hash)
        if cached_chunks:
            return cached_chunks
        
        # Process document
        document_text = await self.circuit_breaker.call(
            self.doc_processor.process_document,
            document_source
        )
        
        # Cache processed document
        cache_manager.set_response_cache("document_chunks", doc_hash, document_text, ttl=3600)
        
        return document_text
    
    @monitor_performance
    async def process_queries_batch(self, 
                                  document_source: str, 
                                  questions: List[str],
                                  client_id: str = "default") -> Dict[str, Any]:
        """Batch process multiple queries with optimizations"""
        
        # Rate limiting check
        if not await self.rate_limiter.is_allowed(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Process document once
        document_text = await self.process_document_optimized(document_source)
        doc_hash = hashlib.md5(document_text.encode()).hexdigest()
        
        # Chunk and embed document
        chunks = self.text_chunker.chunk_text(document_text, "policy_document")
        await self.vector_store.batch_embed_chunks(chunks)
        
        # Process queries concurrently
        tasks = []
        for question in questions:
            task = self._process_single_query_optimized(question, doc_hash)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i}: {result}")
                answers.append(f"Error processing question: {str(result)}")
            else:
                answers.append(result)
        
        return {
            "answers": answers,
            "processing_metadata": {
                "document_hash": doc_hash,
                "chunks_processed": len(chunks),
                "questions_count": len(questions),
                "timestamp": time.time()
            }
        }
    
    async def _process_single_query_optimized(self, question: str, doc_hash: str) -> str:
        """Process single query with all optimizations"""
        
        # Check response cache
        cached_response = cache_manager.get_response_cache(question, doc_hash)
        if cached_response:
            return cached_response
        
        # Parse query
        structured_query = await self.llm_processor.parse_query(question)
        
        # Enhance query with domain context
        enhanced_query = self.query_processor.enhance_query(question, structured_query)
        
        # Search for relevant chunks
        relevant_chunks = self.vector_store.search_similar_chunks(enhanced_query, top_k=5)
        
        # Check for early exit
        if self.early_exit.should_exit_early(structured_query, relevant_chunks):
            # Use top chunk for quick response
            quick_response = await self._generate_quick_response(structured_query, relevant_chunks[:1])
            cache_manager.set_response_cache(question, doc_hash, quick_response, ttl=900)
            return quick_response
        
        # Full analysis
        detailed_response = await self.llm_processor.analyze_and_decide(structured_query, relevant_chunks)
        
        # Cache result
        cache_manager.set_response_cache(question, doc_hash, detailed_response, ttl=1800)
        
        return detailed_response
    
    async def _generate_quick_response(self, 
                                     query: StructuredQuery, 
                                     chunks: List[ClauseMatch]) -> str:
        """Generate quick response for high-confidence cases"""
        if not chunks:
            return "No relevant information found in the document."
        
        top_chunk = chunks[0]
        
        # Simple pattern matching for common queries
        if any(word in query.raw_query.lower() for word in ['covered', 'eligible']):
            if any(word in top_chunk.text.lower() for word in ['covered', 'eligible', 'entitled']):
                return f"Yes, this appears to be covered. Based on: {top_chunk.text[:200]}..."
            else:
                return f"This may not be covered. Based on: {top_chunk.text[:200]}..."
        
        # Fallback to chunk content
        return f"Based on the policy: {top_chunk.text[:300]}..."

# Global optimized system instance
optimized_system = OptimizedDocumentAnalysisSystem()

# Middleware for request/response logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(f"Response: {response.status_code} in {process_time:.2f}s")
    
    return response

# Enhanced endpoint with all optimizations
@app.post("/api/v1/hackrx/run/optimized")
async def process_query_optimized(
    request: QueryRequest,
    client_ip: str = Header(None, alias="X-Forwarded-For"),
    token: str = Depends(verify_token)
):
    """Optimized endpoint with caching, rate limiting, and performance monitoring"""
    
    client_id = client_ip or "unknown"
    
    try:
        result = await optimized_system.process_queries_batch(
            request.documents,
            request.questions,
            client_id
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimized processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Health check with system status
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system status"""
    
    health_status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check Gemini AI
    try:
        if GEMINI_API_KEY and optimized_system.llm_processor.model:
            test_response = await asyncio.wait_for(
                optimized_system.llm_processor.model.generate_content("test"),
                timeout=5.0
            )
            health_status["services"]["gemini"] = "healthy"
        else:
            health_status["services"]["gemini"] = "not_configured"
    except Exception as e:
        health_status["services"]["gemini"] = f"error: {str(e)}"
    
    # Check Pinecone
    try:
        if optimized_system.vector_store.index:
            stats = optimized_system.vector_store.index.describe_index_stats()
            health_status["services"]["pinecone"] = {
                "status": "healthy",
                "total_vectors": stats.total_vector_count
            }
        else:
            health_status["services"]["pinecone"] = "fallback_mode"
    except Exception as e:
        health_status["services"]["pinecone"] = f"error: {str(e)}"
    
    # Check Redis cache
    try:
        if cache_manager.redis_available:
            cache_manager.redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        else:
            health_status["services"]["redis"] = "memory_fallback"
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
    
    # Overall status
    service_statuses = [
        status for status in health_status["services"].values() 
        if isinstance(status, str)
    ]
    
    if any("error" in status for status in service_statuses):
        health_status["status"] = "degraded"
    
    return health_status

# Performance metrics endpoint
@app.get("/api/v1/metrics")
async def get_performance_metrics(token: str = Depends(verify_token)):
    """Get system performance metrics"""
    
    return {
        "cache_stats": {
            "redis_available": cache_manager.redis_available,
            "cache_type": "redis" if cache_manager.redis_available else "memory"
        },
        "rate_limiter": {
            "max_requests": optimized_system.rate_limiter.max_requests,
            "window_seconds": optimized_system.rate_limiter.window_seconds,
            "active_clients": len(optimized_system.rate_limiter.requests)
        },
        "circuit_breaker": {
            "state": optimized_system.circuit_breaker.state,
            "failure_count": optimized_system.circuit_breaker.failure_count
        },
        "timestamp": time.time()
    }