"""
Flask RAG Chatbot Application
Main application file with API endpoints and configuration.
"""

import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
from typing import Dict, Any

# Import our RAG components
from models.document_processor import DocumentProcessor
from models.text_chunker import TextChunker, ChunkConfig
from models.vector_store import RAGVectorStore
from models.deepseek_client import DeepSeekClient, LLMConfig, RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global components
document_processor = None
text_chunker = None
vector_store = None
deepseek_client = None
rag_pipeline = None

def initialize_components():
    """Initialize all RAG components."""
    global document_processor, text_chunker, vector_store, deepseek_client, rag_pipeline
    
    try:
        logger.info("Initializing RAG components...")
        
        # Initialize document processor
        document_processor = DocumentProcessor()
        
        # Initialize text chunker with optimized config
        chunk_config = ChunkConfig(
            chunk_size=600,
            overlap_size=75,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        text_chunker = TextChunker(chunk_config)
        
        # Initialize vector store
        vector_store = RAGVectorStore(model_name='all-MiniLM-L6-v2')
        
        # Initialize DeepSeek client
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or api_key == 'your_openrouter_api_key_here':
            logger.warning("OpenRouter API key not set. LLM functionality will be limited.")
            deepseek_client = None
            rag_pipeline = None
        else:
            llm_config = LLMConfig(
                model="deepseek/deepseek-r1",
                max_tokens=1000,
                temperature=0.7
            )
            deepseek_client = DeepSeekClient(api_key, llm_config)
            rag_pipeline = RAGPipeline(deepseek_client, vector_store)
            
            # Test connection
            connection_test = deepseek_client.test_connection()
            if connection_test['success']:
                logger.info("DeepSeek R1 connection successful")
            else:
                logger.error(f"DeepSeek R1 connection failed: {connection_test['message']}")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Initialize components on startup
initialize_components()

@app.route('/')
def index():
    """Main page."""
    stats = vector_store.get_stats() if vector_store else {}
    return render_template('index.html', stats=stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            return jsonify({'error': 'Only PDF and DOCX files are supported'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        logger.info(f"File uploaded: {filepath}")
        
        # Validate file
        if not document_processor.validate_file(filepath):
            os.remove(filepath)  # Clean up invalid file
            return jsonify({'error': 'File validation failed'}), 400
        
        # Process document
        text = document_processor.process_document(filepath)
        if not text.strip():
            os.remove(filepath)
            return jsonify({'error': 'No text could be extracted from the document'}), 400
        
        # Chunk text
        chunks_with_metadata = text_chunker.chunk_with_metadata(text, filename)
        chunks = [chunk['content'] for chunk in chunks_with_metadata]
        metadata = chunks_with_metadata
        
        # Add to vector store
        vector_store.add_documents(chunks, metadata)
        
        # Clean up uploaded file (optional - remove if you want to keep originals)
        # os.remove(filepath)
        
        response_data = {
            'success': True,
            'message': f'Document processed successfully',
            'filename': filename,
            'chunks_count': len(chunks),
            'char_count': len(text),
            'stats': vector_store.get_stats()
        }
        
        logger.info(f"Document processed: {filename}, {len(chunks)} chunks")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """Handle document queries."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        k = data.get('k', 5)  # Number of chunks to retrieve
        k = max(1, min(k, 20))  # Limit k to reasonable range
        
        if not rag_pipeline:
            return jsonify({'error': 'RAG pipeline not available. Please check your OpenRouter API key.'}), 503
        
        if vector_store.vector_store.index.ntotal == 0:
            return jsonify({'error': 'No documents have been uploaded yet. Please upload a document first.'}), 400
        
        # Process query through RAG pipeline
        result = rag_pipeline.process_query(query, k)
        
        # Format response
        response_data = {
            'query': query,
            'answer': result.get('answer', 'No answer generated'),
            'chunks': result.get('chunks', []),
            'scores': result.get('scores', []),
            'metadata': {
                'model': result.get('model', 'deepseek/deepseek-r1'),
                'chunks_retrieved': len(result.get('chunks', [])),
                'usage': result.get('usage', {}),
                'retrieval_k': k
            }
        }
        
        logger.info(f"Query processed: {query[:50]}...")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear all documents from the vector store."""
    try:
        vector_store.clear()
        logger.info("Vector store cleared")
        return jsonify({'success': True, 'message': 'All documents cleared'})
    except Exception as e:
        logger.error(f"Clear operation error: {e}")
        return jsonify({'error': f'Clear operation failed: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Get current system statistics."""
    try:
        stats = vector_store.get_stats() if vector_store else {}
        
        # Add API status
        api_status = 'connected' if rag_pipeline else 'not_configured'
        if deepseek_client and rag_pipeline:
            test_result = deepseek_client.test_connection()
            api_status = 'connected' if test_result['success'] else 'error'
        
        stats['api_status'] = api_status
        stats['upload_folder'] = app.config['UPLOAD_FOLDER']
        stats['max_file_size_mb'] = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500

@app.route('/test_api')
def test_api():
    """Test the DeepSeek API connection."""
    if not deepseek_client:
        return jsonify({
            'success': False, 
            'message': 'DeepSeek client not configured. Check your API key.'
        })
    
    try:
        result = deepseek_client.test_connection()
        return jsonify(result)
    except Exception as e:
        logger.error(f"API test error: {e}")
        return jsonify({'success': False, 'message': f'Test failed: {str(e)}'})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
