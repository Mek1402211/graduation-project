# AI-Powered Accessibility Backend Server

A Flask-based backend API server designed to empower blind and visually impaired users in their daily life tasks. This server provides intelligent AI-powered features accessed through a mobile app, including real-time currency detection, scene description, text recognition and translation, and voice-controlled command processing.

## Key Features for Daily Living Assistance

### 🎙️ Voice-Controlled Commands
- **Voice Command Recognition**: Process audio commands (WAV, MP3, OGG, FLAC) using Whisper ASR for hands-free operation
- **Intelligent Task Routing**: Automatically understand user intent and route to appropriate task (currency, text, or scene)
- **Bilingual Support**: Full English and Arabic language support for accessibility in multilingual environments
- **Natural Feedback**: Clear audio and text feedback in the user's preferred language

### 💵 Currency Detection & Identification
- **Instant Money Recognition**: Quickly identify currency denominations using camera input
- **Multiple Currency Support**: Detect local and international currencies
- **Perfect for Daily Transactions**: Essential for shopping, banking, and financial independence
- **Real-time Processing**: Fast results for practical daily use

### 🔤 Text Recognition & Translation
- **Optical Character Recognition (OCR)**: Read text from documents, signs, labels, and packages
- **Automatic Translation**: Translate recognized text into preferred languages
- **Everyday Use Cases**: Read product labels, menus, documents, street signs, and more
- **Accuracy Optimized**: Fine-tuned for real-world conditions and various lighting

### 🌍 Scene Description & Environment Awareness
- **Detailed Scene Analysis**: Understand and describe surroundings through AI vision
- **Navigation Assistance**: Identify objects, obstacles, and environmental features
- **Accessibility Information**: Describe locations, help with wayfinding and spatial awareness
- **Enhanced Independence**: Better understanding of surroundings for safe navigation
 & Setup

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for optimal performance)
- **Storage**: 10GB+ for model files
- **Processor**: Multi-core processor (GPU optional but recommended)
- **OS**: Linux, macOS, or Windows

### Prerequisites
- pip package manager
- Tesseract OCR engine
- Git (for version control)

### Quick Start Guide

1. **Clone or download the project**
   ```bash
   cd graduation-project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Credentials**
   - Obtain a Google Generative AI API key from [Google AI Studio](https://aistudio.google.com/)
   - Add your key to `utils/apikey.txt`:
     ```
     your-google-api-key-here
     ```
   - (Optional) Create `utils/.env` for additional configuration

5. **Verify Pre-trained Models**
   - Ensure `voice_command_model/` contains the voice classification model
   - Ensure `models/project_model1.pt` contains the PyTorch model
   - Download models if not included in the repository

6. **Test Installation**
   ```bash
   python app.py
   # Server should start on http://localhost:5000
   ``
2. **Install dependencies**
   ```bash
   Mobile App Integration - API Endpoints

The mobile application communicates with these endpoints to provide accessibility features:

### 🏥 Health Check
**`GET /`** - Verify server is running
- **Response**: `{"status": "active"}`
- **Use Case**: Mobile app startup verification

### 💰 Currency Detection
**`POST /currency`** - Identify money denominations
- **Input**: Image file (PNG, JPG, JPEG)
- **Parameter**: `image` (multipart form-data)
- **Response**: `{"currencys": ["USD 100", "EUR 50", ...]}`
- **Use Case**: Help blind users identify and verify currency denominations

### 📄 Text Recognition & Translation
**`POST /ocr-translate`** - Read and translate text
- **Input**: Image file containing text
- **Parameter**: `image` (multipart form-data)
- **Response**: Recognized text with translations
- **Use Case**: Read labels, documents, signs, menus, product information

### 🎙️ Voice Command Processing
**`POST /api/voice-command`** - Process voice commands
- **Input**: Audio file (WAV, MP3, OGG, FLAC)
- **Parameter**: `audio` (multipart form-data)
- **Max SizIntegration (Mobile Backend)
```python
import requests

# Example 1: Currency Detection
def detect_money(image_path, server_url='http://localhost:5000'):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{server_url}/currency', files=files)
        currencies = response.json().get('currencys')
        return currencies  # e.g., ["USD 100", "EUR 50"]

# Example 2: Voice Command Processing
def process_voice_command(audio_path, server_url='http://localhost:5000'):
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(f'{server_url}/api/voice-command', files=files)
        result = response.json()
        return result  # command, task, description
API Keys & Credentials
Edit `utils/apikey.txt`:
```
your-google-generativeai-key-here
```

### Environment Variables (Optional)
Create `utils/.env` for additional settings:
```
GOOGLE_API_KEY=your_google_api_key_here
DEBUG=False
LOG_LEVEL=INFO
```

### Server Configuration
- **Upload Directory**: `uploads/` (auto-created)
- **Maximum File Size**: 16MB (configurable in app.py)
- **Supported File Types**:
  - **Images**: png, jpg, jpeg (for vision tasks)
  - **Audio**: wav, mp3, ogg, flac (for voice commands)
  - **Video**: mp4 (future feature)

### Performance Optimization
- **Model Caching**: Models are loaded once at startup
- **Temporary File Cleanup**: Audio files are automatically deleted after processing
- **Multi-threading**: Flask handles concurrent requests
- *Deployment

### Development Server
Perfect for testing and development:
```bash
python app.py
```
- Server: `http://localhost:5000`
- Hot reload: Not enabled (restart manually)
- Debugging: Full error messages in console

### Production Deployment
For mobile app users and real-world deployment:

#### Using Gunicorn (Linux/macOS)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Waitress (Windows/Cross-platform)
```bash
pip install waitress
waitress-serve --port=5000 app:app
```

#### Docker Deployment (Recommended)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### AWS Deployment
- **EC2**: Deploy using Gunicorn with Nginx reverse proxy
- **Elastic Beanstalk**: Auto-scaling Flask application
- **Lambda**: Serverless option with API Gateway (requires refactoring)
- **API Gateway**: For secure HTTPS endpoints

### Environment Variables for Production
```bash
export FLASK_ENV=production
export DEBUG=False
export WORKERS=4
export THREADS=2
curl http://localhost:5000/
```

### Mobile App Integration (Example Flow)
1. User speaks: "What's in front of me?" 
2. Mobile app captures voice → sends to `/api/voice-command`
3. Server identifies intent → routes to scene description
4. Mobile app receives description → speaks it back to user
5. Important Notes for Blind Users & Developers

### Accessibility Features
✅ **Voice-First Design**: All major features accessible via voice commands  
✅ **Bilingual Support**: English and Arabic for diverse user base  
✅ **Fast Processing**: Optimized for quick feedback and response  
✅ **Reliable Offline Fallback**: Graceful error handling and user feedback  
✅ **Battery Efficient**: Optimized for mobile device performance  

### Performance Considerations
- Model files require 5-10GB of disk space
- Minimum 4GB RAM for smooth operation
- First startup may take 30-60 seconds as models are loaded
- Subsequent requests are faster due to model caching
- GPU acceleration (CUDA) significantly improves speed

### Security & Privacy
⚠️ **No Image Storage**: Images are processed and immediately discarded  
⚠️ **No Audio Logging**: Voice commands are not recorded or stored  
⚠️ **API Key Protection**: Never commit API keys to version control  
⚠️ **HTTPS Recommended**: Use SSL/TLS in production  

### Troubleshooting
| Issue | Solution |
|-------|----------|
| "Model not found" | Verify models exist in `voice_command_model/` and `models/` |
| "API key error" | Check `utils/apikey.txt` has valid Google API key |
| Slow response time | Enable GPU support or reduce image resolution |
| "Permission denied" | Run with appropriate user permissions, check file ownership |
| Out of memory | Reduce concurrent requests or increase system RAM |

## Future Enhancements

- 🎬 Real-time video streaming support
- 📊 User activity logging and analytics
- 🔐 Authentication and user management
- 💾 Database integration for user preferences
- 🌐 Multi-language support expansion
- ⚡ Real-time WebSocket for instant notifications
- 📱 Mobile app-specific optimizations
- 🚀 Batch processing for efficiency
- 🎯 Advanced caching and CDN integration

## Support & Contributions

### Getting Help
- Check troubleshooting section above
- Review error messages in console logs
- Test endpoints with cURL before integrating in mobile app

### Contributing
- Report bugs with detailed logs
- Suggest features for accessibility improvements
- Submit pull requests with test cases

## License

[Add your license information here]

## About This Project

This backend server is part of a mission to **empower blind and visually impaired individuals** with technology that enhances their independence and daily living capabilities. By combining state-of-the-art computer vision and voice recognition, we enable users to:

- 💵 **Identify money** with confidence in any transaction
- 📖 **Read documents and signs** instantly
- 🌍 **Understand their environment** better for safe navigation
- 🎙️ **Control everything with voice** for hands-free operation

Your contributions make a real difference in people's lives. ♿❤️
  - Response: Extracted and translated text

### Voice Command Processing
- **`POST /api/voice-command`**
  - Accepts: Audio file (WAV, MP3, OGG, FLAC)
  - Parameter: `audio` (multipart form file)
  - Max file size: 16MB
  - Response: Recognized command and corresponding task

## Usage Examples

### Python Request (Currency Detection)
```python
import requests

# Currency detection
with open('currency_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/currency', files=files)
    print(response.json())
```

### cURL (Voice Command)
```bash
curl -X POST http://localhost:5000/api/voice-command \
  -F "audio=@voice_command.wav"
```

## Configuration

### Environment Variables
Create or edit `utils/.env`:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### Upload Settings
- Maximum file size: 16MB (configurable in code)
- Allowed file extensions:
  - Images: png, jpg, jpeg
  - Audio: wav, mp3, ogg, flac
  - Video: mp4
  - Other: txt

## Running the Server

### Development Mode
```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

### Production Mode (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Dependencies

- **flask** - Web framework
- **werkzeug** - HTTP utilities
- **pytesseract** - OCR interface
- **opencv-python** - Image processing
- **pillow** - Image manipulation
- **googletrans** - Language translation
- **ultralytics** - YOLO object detection
- **transformers** - HuggingFace models
- **torch** - Deep learning framework
- **google-generativeai** - Google AI integration
- **python-dotenv** - Environment variable management
- **numpy** - Numerical computing

See `requirements.txt` for complete list with versions.

## Key Classes & Modules

### VoiceCommandInference
Located in `voice_inference.py`, handles:
- Loading Whisper and voice command classification models
- Processing audio files
- Mapping voice commands to tasks
- Bilingual task descriptions

### Utility Modules
- **`api_gminie.py`**: Integration with Google Generative AI for scene descriptions
- **`currency.py`**: Currency detection and classification
- **`ocr_translate.py`**: OCR with automatic translation
- **`apikey.txt`**: Stores API credentials

## Error Handling

The API includes comprehensive error handling:
- Missing file validation
- File type validation
- File size limits
- Graceful exception messages
- Automatic temporary file cleanup

## Notes

- Uploaded files are temporarily stored in `uploads/` directory
- Voice processing files are cleaned up automatically after processing
- Ensure adequate disk space for model files (~5-10GB depending on model sizes)
- GPU support available with CUDA-enabled PyTorch installation

## Future Enhancements

- Real-time streaming audio support
- Batch processing for multiple files
- Advanced caching mechanisms
- Database integration for user history
- Containerization with Docker
- API authentication and rate limiting

## Support

For issues, questions, or contributions, please contact the development team.


