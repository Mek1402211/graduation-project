import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import asyncio
#from utils.distance import estimate_distance
from utils.currency import detect_currency
from utils.ocr_translate import ocr_and_translate
#from utils.object_color import detect_objects_and_colors
#from utils.scene_description import describe_scene
from utils.api_gminie import describe_scene2
from utils.api_gminie import reconsteuct_to_arabice
import logging
from voice_inference import VoiceCommandInference  # Your existing class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','mp4',"txt",'wav', 'mp3', 'ogg', 'flac'}
inference_system = VoiceCommandInference()
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/', methods=['GET'])
def isactive():
    return jsonify({"status": "active"}), 200

@app.route('/currency', methods=['POST'])
def currency_detection():
   try:
      if 'image' not in request.files:
          return jsonify({"error": "No image uploaded"}), 400
      file = request.files['image']
      if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            currency = detect_currency(filepath)
            return jsonify({"currencys":currency}),{'Content-Type': 'application/json; charset=utf-8'}
   except Exception as e:
           print(f"Error in currency detection: {e}")
           return jsonify({"error": str(e)}), 500

@app.route('/ocr-translate', methods=['POST'])
def ocr_translation():
        try:
           if 'image' not in request.files: 
              return jsonify({"error": "No image uploaded"}), 400
           file = request.files['image']
           if file and allowed_file(file.filename):
              filename = secure_filename(file.filename)
           filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
           file.save(filepath)
           translated_text = asyncio.run(ocr_and_translate(filepath))
           reconstructed_text = reconsteuct_to_arabice(translated_text)
           return jsonify({'translated_text': str(reconstructed_text)}), {'Content-Type': 'application/json; charset=utf-8'}
        except Exception as e:
              print(f"Error in ocr-translate: {e}")
              return jsonify({"error": str(e)}), 500

@app.route('/describe-scene2', methods=['POST'])
def scene_description2():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            description = describe_scene2(filepath)
            return jsonify({"description": description})
    except Exception as e:
        print(f"Error in scene description: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice-command', methods=['POST'])
def process_voice_command():
    """Endpoint for processing voice commands"""
    # Check if file was uploaded
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    # Validate file
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        # Process the audio
        result = inference_system.process_voice_command(temp_path)
        # Clean up
        os.remove(temp_path)
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400  
    except Exception as e:
        logging.error(f"Error processing voice command: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500
    
@app.route('/api/text-command', methods=['POST'])
def process_text_command():
    """Endpoint for processing text commands"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        language = data.get('language', 'auto')
        result = inference_system.process_text_command(data['text'], language)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error processing text command: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500
    
@app.route('/api/supported-commands', methods=['GET'])
def get_supported_commands():
    """Endpoint to list supported commands"""
    try:
        commands = inference_system.get_supported_commands()
        return jsonify(commands), 200
    except Exception as e:
        logging.error(f"Error getting supported commands: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)