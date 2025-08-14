# RAG Chatbot - Local Flask Application

A modern, clean RAG (Retrieval-Augmented Generation) chatbot built with Flask and DeepSeek R1.

## Features

✅ **Modern UI/UX**: Clean, subtle, and uniform design  
✅ **Document Support**: PDF and DOCX file ingestion  
✅ **Smart Chunking**: Recursive character splitting with overlap  
✅ **Semantic Search**: FAISS vector store with Sentence Transformers  
✅ **DeepSeek R1 Integration**: Powered by OpenRouter API  
✅ **Real-time Chat**: Interactive Q&A interface  
✅ **Error Handling**: Comprehensive logging and user feedback  

## Architecture

```
├── app.py                 # Flask application
├── models/               # Core RAG components
│   ├── document_processor.py    # PDF/DOCX ingestion
│   ├── text_chunker.py          # Text chunking strategies
│   ├── vector_store.py          # FAISS + embeddings
│   └── deepseek_client.py       # DeepSeek R1 integration
├── static/              # Frontend assets
│   ├── css/style.css           # Modern CSS styling
│   └── js/app.js              # Interactive JavaScript
├── templates/           # HTML templates
└── uploads/            # Document storage
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env` file and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_actual_openrouter_api_key_here
```

### 3. Run the Application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Usage

1. **Upload Documents**: Drag & drop or click to upload PDF/DOCX files
2. **Ask Questions**: Type questions about your uploaded documents
3. **Get Answers**: Receive AI-generated responses with source citations

## Configuration

### Environment Variables (.env)
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `FLASK_ENV`: Set to `development` for debug mode
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)

### Chunking Strategy
The application uses **Recursive Character Splitting** with:
- **Chunk Size**: 600 characters
- **Overlap**: 75 characters
- **Separators**: Paragraph → Sentence → Word boundaries

This ensures optimal semantic coherence while maintaining context.

### DeepSeek R1 Configuration
- **Model**: `deepseek/deepseek-r1`
- **Max Tokens**: 1000 per response
- **Temperature**: 0.7 for balanced creativity
- **Context Limit**: 8000 tokens

## API Endpoints

- `POST /upload` - Upload and process documents
- `POST /query` - Ask questions about documents  
- `POST /clear` - Clear all documents
- `GET /stats` - Get system statistics
- `GET /test_api` - Test DeepSeek API connection

## Development

### Project Structure
```
RAG Flask Application
├── 📁 models/           # Core business logic
├── 📁 static/           # Frontend assets  
├── 📁 templates/        # Jinja2 templates
├── 📁 uploads/          # File storage
├── 📁 logs/             # Application logs
└── 📄 app.py           # Main Flask app
```

### Key Components

**Document Processing**
- PDF extraction with PyPDF2
- DOCX parsing with python-docx
- Text cleaning and validation

**Text Chunking**  
- Recursive character splitting
- Semantic boundary preservation
- Configurable overlap strategy

**Vector Storage**
- FAISS similarity search
- Sentence Transformers embeddings
- Metadata preservation

**LLM Integration**
- OpenRouter API client
- DeepSeek R1 model
- Context-aware prompting

## Troubleshooting

### Common Issues

**API Connection Failed**
- Verify your OpenRouter API key in `.env`
- Check internet connection
- Ensure API key has proper permissions

**File Upload Errors**
- Check file size (max 16MB)
- Ensure file is PDF or DOCX format
- Verify file is not corrupted

**No Search Results**
- Confirm documents are uploaded
- Try broader or more specific queries
- Check if text was extracted properly

### Logs
Application logs are stored in `logs/app.log` for debugging.

## License

MIT License - feel free to modify and use for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Built with ❤️ using Flask, DeepSeek R1, and modern web technologies.**
