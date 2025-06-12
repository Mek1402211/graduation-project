import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
import whisper
import re
import json
from datetime import datetime
import logging

class VoiceCommandInference:
    def __init__(self, model_path="./voice_command_model", whisper_model_size="base"):
        """
        Initialize the voice command inference system
        
        Args:
            model_path: Path to trained model
            whisper_model_size: Whisper model size (base, small, medium, large)
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.whisper_model = None
        
        # Task mapping
        self.task_labels = {
            0: "scene_description",
            1: "currency_detection", 
            2: "ocr_translation"
        }
        
        # Task descriptions for user feedback
        self.task_descriptions = {
            "scene_description": {
                "en": "I'll describe what I see in the image",
                "ar": "سأصف ما أراه في الصورة"
            },
            "currency_detection": {
                "en": "I'll identify the currency in the image", 
                "ar": "سأتعرف على العملة في الصورة"
            },
            "ocr_translation": {
                "en": "I'll read and translate the text in the image",
                "ar": "سأقرأ وأترجم النص في الصورة"
            }
        }
        
        # Load models
        self.load_models(whisper_model_size)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_models(self, whisper_model_size):
        """Load all required models"""
        try:
            # Load Whisper for speech-to-text
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model(whisper_model_size)
            
            # Load trained classification model
            print(f"Loading trained model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()  # Set to evaluation mode
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def preprocess_text(self, text, language=None):
        """Preprocess text for classification"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle Arabic text
        if language == 'ar' or self._is_arabic(text):
            # Remove diacritics
            text = re.sub(r'[ًٌٍَُِّْ]', '', text)
            # Normalize Arabic characters
            text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
            text = text.replace('ة', 'ه').replace('ى', 'ي')
        
        return text.lower()

    def _is_arabic(self, text):
        """Check if text contains Arabic characters"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(text))

    def speech_to_text(self, audio_file_path):
        """Convert speech to text using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "confidence": result.get("avg_logprob", 0.0)
            }
        except Exception as e:
            self.logger.error(f"Speech-to-text error: {e}")
            return None

    def classify_command(self, text, language=None):
        """Classify the voice command"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text, language)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions, dim=-1)[0].item()
            
            return {
                "task": self.task_labels[predicted_class],
                "task_id": predicted_class,
                "confidence": confidence,
                "processed_text": processed_text
            }
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return None

    def process_voice_command(self, audio_file_path):
        """Complete pipeline: audio -> text -> task classification"""
        
        # Step 1: Speech to text
        stt_result = self.speech_to_text(audio_file_path)
        if not stt_result:
            return {
                "success": False,
                "error": "Failed to convert speech to text"
            }
        
        # Step 2: Classify command
        classification = self.classify_command(
            stt_result["text"], 
            stt_result["language"]
        )
        
        if not classification:
            return {
                "success": False,
                "error": "Failed to classify command"
            }
        
        # Step 3: Prepare response
        response = {
            "success": True,
            "transcribed_text": stt_result["text"],
            "detected_language": stt_result["language"],
            "task": classification["task"],
            "task_id": classification["task_id"],
            "confidence": classification["confidence"],
            "task_description": self.get_task_description(
                classification["task"], 
                stt_result["language"]
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        return response

    def get_task_description(self, task, language):
        """Get user-friendly task description"""
        descriptions = self.task_descriptions.get(task, {})
        return descriptions.get(language, descriptions.get("en", f"Execute {task}"))

    def process_text_command(self, text, language="auto"):
        """Process text command directly (for testing)"""
        
        # Detect language if auto
        if language == "auto":
            language = "ar" if self._is_arabic(text) else "en"
        
        # Classify command
        classification = self.classify_command(text, language)
        
        if not classification:
            return {
                "success": False,
                "error": "Failed to classify command"
            }
        
        return {
            "success": True,
            "input_text": text,
            "detected_language": language,
            "task": classification["task"],
            "task_id": classification["task_id"],
            "confidence": classification["confidence"],
            "task_description": self.get_task_description(
                classification["task"], 
                language
            )
        }

    def get_supported_commands(self):
        """Return examples of supported commands"""
        return {
            "scene_description": {
                "english": ["describe the scene", "what do you see", "tell me what's in the picture"],
                "arabic": ["وصف المشهد", "ماذا ترى", "اخبرني ماذا في الصورة"]
            },
            "currency_detection": {
                "english": ["detect currency", "what currency is this", "identify money"],
                "arabic": ["كشف العملة", "ما هذه العملة", "تحديد الأموال"]
            },
            "ocr_translation": {
                "english": ["translate the page", "read and translate", "scan text"],
                "arabic": ["ترجم الصفحة", "اقرأ وترجم", "امسح النص"]
            }
        }

# Example usage and testing functions
class VoiceCommandTester:
    def __init__(self, inference_system):
        self.inference = inference_system

    def test_text_commands(self):
        """Test with various text commands"""
        test_cases = [
            ("describe the scene", "en"),
            ("وصف المشهد", "ar"),
            ("what currency is this", "en"),
            ("ما هذه العملة", "ar"),
            ("translate the page", "en"),
            ("ترجم الصفحة", "ar"),
            ("please tell me what you see", "en"),
            ("من فضلك اخبرني ماذا ترى", "ar"),
            ("can you detect the money", "en"),
            ("هل يمكنك كشف الأموال", "ar"),
            ("I want to translate this text", "en"),
            ("أريد أن أترجم هذا النص", "ar")
        ]

        print("Testing Voice Command Classification")
        print("=" * 50)

        for text, expected_lang in test_cases:
            result = self.inference.process_text_command(text)
            
            if result["success"]:
                print(f"Input: '{text}'")
                print(f"Language: {result['detected_language']}")
                print(f"Task: {result['task']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Description: {result['task_description']}")
                print("-" * 30)
            else:
                print(f"Failed to process: '{text}' - {result['error']}")

    def test_edge_cases(self):
        """Test edge cases and potential issues"""
        edge_cases = [
            ("", "en"),  # Empty text
            ("hello", "en"),  # Irrelevant text
            ("مرحبا", "ar"),  # Irrelevant Arabic text
            ("um describe the scene please", "en"),  # With filler words
            ("يعني وصف المشهد من فضلك", "ar"),  # With Arabic filler
            ("DESCRIBE THE SCENE", "en"),  # All caps
            ("وصف المشهد!!", "ar"),  # With punctuation
        ]

        print("\nTesting Edge Cases")
        print("=" * 30)

        for text, lang in edge_cases:
            result = self.inference.process_text_command(text, lang)
            
            print(f"Input: '{text}'")
            if result["success"]:
                print(f"Task: {result['task']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"Error: {result['error']}")
            print("-" * 20)

    def benchmark_performance(self, num_tests=100):
        """Benchmark classification performance"""
        import time
        
        test_commands = [
            "describe the scene",
            "وصف المشهد", 
            "detect currency",
            "كشف العملة",
            "translate page",
            "ترجم الصفحة"
        ]
        
        print(f"\nBenchmarking Performance ({num_tests} tests)")
        print("=" * 40)
        
        start_time = time.time()
        
        for i in range(num_tests):
            command = test_commands[i % len(test_commands)]
            result = self.inference.process_text_command(command)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_tests
        
        print(f"Average processing time: {avg_time:.4f} seconds")
        print(f"Throughput: {1/avg_time:.1f} commands/second")

def main():
    """Main function to run the inference system"""
    try:
        # Initialize inference system
        print("Initializing Voice Command Inference System...")
        inference = VoiceCommandInference()
        
        # Create tester
        tester = VoiceCommandTester(inference)
        
        # Run tests
        tester.test_text_commands()
        tester.test_edge_cases()
        tester.benchmark_performance()
        
        # Show supported commands
        print("\nSupported Commands:")
        print("=" * 30)
        commands = inference.get_supported_commands()
        for task, examples in commands.items():
            print(f"\n{task.upper()}:")
            print("English:", ", ".join(examples["english"][:3]))
            print("Arabic:", ", ".join(examples["arabic"][:3]))
        
        print("\nInference system ready for use!")
        
        # Interactive mode (optional)
        print("\nEnter 'interactive' to test commands manually, or 'quit' to exit:")
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'interactive':
                interactive_mode(inference)
            elif user_input:
                result = inference.process_text_command(user_input)
                if result["success"]:
                    print(f"Task: {result['task']} (confidence: {result['confidence']:.3f})")
                    print(f"Description: {result['task_description']}")
                else:
                    print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"Error initializing system: {e}")

def interactive_mode(inference):
    """Interactive testing mode"""
    print("\nInteractive Mode - Enter voice commands to test")
    print("Type 'back' to return to main menu")
    print("-" * 40)
    
    while True:
        command = input("Enter command: ").strip()
        
        if command.lower() == 'back':
            break
        elif command:
            result = inference.process_text_command(command)
            if result["success"]:
                print(f"✓ Task: {result['task']}")
                print(f"  Language: {result['detected_language']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Action: {result['task_description']}")
            else:
                print(f"✗ Error: {result['error']}")

if __name__ == "__main__":
    main()