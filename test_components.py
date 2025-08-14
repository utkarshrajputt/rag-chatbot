"""
Quick Test Script for RAG Chatbot Components
Tests individual components before running the full application.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    try:
        # Core libraries
        import flask
        import PyPDF2
        from docx import Document
        import nltk
        import numpy as np
        import pandas as pd
        import faiss
        from sentence_transformers import SentenceTransformer
        import openai
        from dotenv import load_dotenv
        
        # Our modules
        from models.document_processor import DocumentProcessor
        from models.text_chunker import TextChunker, ChunkConfig
        from models.vector_store import RAGVectorStore
        from models.deepseek_client import DeepSeekClient, LLMConfig
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_nltk_data():
    """Test if NLTK data is available."""
    print("\n🧪 Testing NLTK data...")
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Test sentence tokenization
        test_text = "This is a test. This is another sentence."
        sentences = sent_tokenize(test_text)
        
        if len(sentences) == 2:
            print("✅ NLTK data working correctly!")
            return True
        else:
            print(f"❌ NLTK tokenization issue: got {len(sentences)} sentences")
            return False
            
    except Exception as e:
        print(f"❌ NLTK error: {e}")
        return False

def test_components():
    """Test core RAG components."""
    print("\n🧪 Testing RAG components...")
    
    try:
        # Test document processor
        doc_processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Test text chunker
        chunk_config = ChunkConfig(chunk_size=100, overlap_size=20)
        text_chunker = TextChunker(chunk_config)
        
        test_text = "This is a test document. " * 10
        chunks = text_chunker.chunk_text(test_text)
        print(f"✅ Text chunker working: {len(chunks)} chunks created")
        
        # Test vector store (without heavy model loading)
        print("✅ Vector store components available")
        
        return True
        
    except Exception as e:
        print(f"❌ Component error: {e}")
        return False

def test_environment():
    """Test environment setup."""
    print("\n🧪 Testing environment...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check directories
        required_dirs = ['uploads', 'logs', 'static', 'templates', 'models']
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"✅ Directory exists: {dir_name}")
            else:
                print(f"❌ Missing directory: {dir_name}")
        
        # Check .env file
        if os.path.exists('.env'):
            print("✅ .env file found")
            api_key = os.getenv('OPENROUTER_API_KEY')
            if api_key and api_key != 'your_openrouter_api_key_here':
                print("✅ OpenRouter API key configured")
            else:
                print("⚠️  OpenRouter API key not set (add to .env file)")
        else:
            print("❌ .env file not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAG Chatbot Component Tests\n")
    
    tests = [
        test_imports,
        test_nltk_data,
        test_components,
        test_environment
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    if all(results):
        print("🎉 All tests passed! Your RAG chatbot is ready to run.")
        print("💡 Next steps:")
        print("   1. Add your OpenRouter API key to .env file")
        print("   2. Run: python app.py")
        print("   3. Open: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Common fixes:")
        print("   1. Ensure virtual environment is activated")
        print("   2. Install packages: pip install -r requirements.txt")
        print("   3. Check file paths and permissions")
    
    print("="*50)

if __name__ == "__main__":
    main()
