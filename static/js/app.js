// RAG Chatbot JavaScript - Clean, Modern Implementation

class RAGChatbot {
    constructor() {
        this.elements = {
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            uploadProgress: document.getElementById('uploadProgress'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            chatMessages: document.getElementById('chatMessages'),
            queryInput: document.getElementById('queryInput'),
            sendBtn: document.getElementById('sendBtn'),
            clearBtn: document.getElementById('clearBtn'),
            resultsCount: document.getElementById('resultsCount'),
            docCount: document.getElementById('docCount'),
            apiStatus: document.getElementById('apiStatus'),
            toastContainer: document.getElementById('toastContainer')
        };
        
        this.currentUpload = null;
        this.isUploading = false;
        this.isQuerying = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAPIStatus();
        this.updateStats();
        
        // Initial UI state
        this.updateChatInputState();
    }

    setupEventListeners() {
        // File upload events
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Drag and drop events
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });

        this.elements.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
        });

        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            
            const files = Array.from(e.dataTransfer.files);
            const validFile = files.find(file => 
                file.type === 'application/pdf' || 
                file.name.toLowerCase().endsWith('.docx')
            );
            
            if (validFile) {
                this.handleFileUpload(validFile);
            } else {
                this.showToast('Please upload a PDF or DOCX file', 'error');
            }
        });

        // Chat events
        this.elements.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleQuery();
            }
        });

        this.elements.sendBtn.addEventListener('click', () => {
            this.handleQuery();
        });

        this.elements.clearBtn.addEventListener('click', () => {
            this.handleClearDocuments();
        });

        // Input state management
        this.elements.queryInput.addEventListener('input', () => {
            this.updateSendButtonState();
        });
    }

    async checkAPIStatus() {
        try {
            const response = await fetch('/test_api');
            const result = await response.json();
            
            const statusElement = this.elements.apiStatus;
            const icon = statusElement.querySelector('i');
            
            if (result.success) {
                statusElement.className = 'stat-value status-indicator connected';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
            } else {
                statusElement.className = 'stat-value status-indicator error';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> API Error';
                this.showToast('DeepSeek API connection failed', 'error');
            }
        } catch (error) {
            const statusElement = this.elements.apiStatus;
            statusElement.className = 'stat-value status-indicator error';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
            console.error('API status check failed:', error);
        }
    }

    async updateStats() {
        try {
            const response = await fetch('/stats');
            const stats = await response.json();
            
            this.elements.docCount.textContent = stats.total_vectors || 0;
            
            // Update chat input state based on document count
            this.updateChatInputState(stats.total_vectors > 0);
            
        } catch (error) {
            console.error('Failed to update stats:', error);
        }
    }

    updateChatInputState(hasDocuments = false) {
        const hasApiConnection = this.elements.apiStatus.classList.contains('connected');
        const canQuery = hasDocuments && hasApiConnection && !this.isQuerying;
        
        this.elements.queryInput.disabled = !canQuery;
        this.elements.sendBtn.disabled = !canQuery;
        
        if (!hasDocuments) {
            this.elements.queryInput.placeholder = "Upload a document first...";
        } else if (!hasApiConnection) {
            this.elements.queryInput.placeholder = "API connection required...";
        } else {
            this.elements.queryInput.placeholder = "Ask a question about your documents...";
        }
    }

    updateSendButtonState() {
        const hasQuery = this.elements.queryInput.value.trim().length > 0;
        const canSend = hasQuery && !this.elements.queryInput.disabled && !this.isQuerying;
        this.elements.sendBtn.disabled = !canSend;
    }

    async handleFileUpload(file) {
        if (this.isUploading) return;
        
        // Validate file
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            this.showToast('File too large. Maximum size is 16MB.', 'error');
            return;
        }

        const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        const validExtensions = ['.pdf', '.docx'];
        const isValidType = validTypes.includes(file.type) || 
                           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!isValidType) {
            this.showToast('Only PDF and DOCX files are supported.', 'error');
            return;
        }

        this.isUploading = true;
        this.showUploadProgress(true);
        this.setProgress(0, 'Preparing upload...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const xhr = new XMLHttpRequest();

            // Progress tracking
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentage = (e.loaded / e.total) * 50; // Upload is 50% of total
                    this.setProgress(percentage, 'Uploading...');
                }
            });

            // Response handling
            xhr.onload = () => {
                if (xhr.status === 200) {
                    this.setProgress(75, 'Processing document...');
                    
                    setTimeout(() => {
                        const result = JSON.parse(xhr.responseText);
                        this.setProgress(100, 'Complete!');
                        
                        setTimeout(() => {
                            this.showUploadProgress(false);
                            this.isUploading = false;
                            this.showToast(`Document processed: ${result.chunks_count} chunks created`, 'success');
                            this.updateStats();
                            this.clearWelcomeMessage();
                            this.elements.fileInput.value = '';
                        }, 1000);
                    }, 500);
                } else {
                    const error = JSON.parse(xhr.responseText);
                    throw new Error(error.error || 'Upload failed');
                }
            };

            xhr.onerror = () => {
                throw new Error('Network error during upload');
            };

            xhr.open('POST', '/upload');
            xhr.send(formData);

        } catch (error) {
            this.showUploadProgress(false);
            this.isUploading = false;
            this.showToast(`Upload failed: ${error.message}`, 'error');
            console.error('Upload error:', error);
        }
    }

    async handleQuery() {
        if (this.isQuerying) return;
        
        const query = this.elements.queryInput.value.trim();
        if (!query) return;

        this.isQuerying = true;
        this.elements.queryInput.disabled = true;
        this.elements.sendBtn.disabled = true;
        this.elements.sendBtn.innerHTML = '<div class="loading"></div>';

        // Add user message
        this.addMessage(query, 'user');
        this.elements.queryInput.value = '';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    k: parseInt(this.elements.resultsCount.value)
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.addMessage(result.answer, 'assistant', {
                    chunks: result.chunks,
                    metadata: result.metadata,
                    scores: result.scores
                });
            } else {
                throw new Error(result.error || 'Query failed');
            }

        } catch (error) {
            this.addMessage('Sorry, I encountered an error processing your question. Please try again.', 'assistant', null, true);
            this.showToast(`Query failed: ${error.message}`, 'error');
            console.error('Query error:', error);
        } finally {
            this.isQuerying = false;
            this.elements.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
            this.updateChatInputState(parseInt(this.elements.docCount.textContent) > 0);
            this.updateSendButtonState();
        }
    }

    async handleClearDocuments() {
        if (confirm('Are you sure you want to clear all documents? This action cannot be undone.')) {
            try {
                const response = await fetch('/clear', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    this.showToast('All documents cleared', 'success');
                    this.clearChat();
                    this.updateStats();
                } else {
                    throw new Error(result.error || 'Clear failed');
                }
                
            } catch (error) {
                this.showToast(`Clear failed: ${error.message}`, 'error');
                console.error('Clear error:', error);
            }
        }
    }

    addMessage(content, type, metadata = null, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type} fade-in`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        if (isError) {
            contentDiv.style.borderLeft = '4px solid var(--error-color)';
        }

        messageDiv.appendChild(contentDiv);

        // Add metadata for assistant messages
        if (type === 'assistant' && metadata && !isError) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = `Retrieved ${metadata.chunks_retrieved || 0} chunks • ${metadata.model || 'DeepSeek R1'}`;
            messageDiv.appendChild(metaDiv);

            // Add sources if available
            if (metadata && Array.isArray(metadata.chunks) && metadata.chunks.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'message-sources';
                
                const title = document.createElement('div');
                title.className = 'source-title';
                title.textContent = 'Source Chunks:';
                sourcesDiv.appendChild(title);

                metadata.chunks.forEach((chunk, index) => {
                    const sourceDiv = document.createElement('div');
                    sourceDiv.className = 'source-item';
                    const preview = chunk.length > 100 ? chunk.substring(0, 100) + '...' : chunk;
                    const score = metadata.scores && metadata.scores[index] ? 
                                 ` (relevance: ${(metadata.scores[index] * 100).toFixed(1)}%)` : '';
                    sourceDiv.textContent = `${index + 1}. ${preview}${score}`;
                    sourcesDiv.appendChild(sourceDiv);
                });

                messageDiv.appendChild(sourcesDiv);
            }
        }

        this.elements.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    clearWelcomeMessage() {
        const welcomeMessage = this.elements.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
    }

    clearChat() {
        this.elements.chatMessages.innerHTML = `
            <div class="welcome-message">
                <i class="fas fa-info-circle"></i>
                <p>Upload a document to get started, then ask questions about its content!</p>
            </div>
        `;
    }

    scrollToBottom() {
        // Use requestAnimationFrame for smooth scrolling
        requestAnimationFrame(() => {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        });
    }

    showUploadProgress(show) {
        this.elements.uploadProgress.style.display = show ? 'block' : 'none';
    }

    setProgress(percentage, text) {
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = text;
    }

    showToast(message, type = 'info', duration = 4000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = this.getToastIcon(type);
        toast.innerHTML = `<i class="${icon}"></i><span>${message}</span>`;
        
        this.elements.toastContainer.appendChild(toast);
        
        // Auto remove
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, duration);
        
        // Manual close on click
        toast.addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }

    getToastIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGChatbot();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        // Refresh stats when page becomes visible
        setTimeout(() => {
            if (window.ragChatbot) {
                window.ragChatbot.updateStats();
                window.ragChatbot.checkAPIStatus();
            }
        }, 500);
    }
});
