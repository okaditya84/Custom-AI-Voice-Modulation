# Custom AI Voice Modulation

A comprehensive pipeline for AI-powered voice modulation and persona creation. This project combines speech recognition, natural language processing, and speech synthesis to create personalized voice responses with defined character attributes.

## 🌟 Features
- Speech-to-Text: Converts spoken audio to text using state-of-the-art Whisper ASR model
- Persona-based Text Generation: Transforms text into responses matching specific character personalities
- High-Quality Text-to-Speech: Converts modified text back to natural-sounding speech
- Audio Processing: Handles different audio formats and combines multiple audio segments
- Complete Pipeline: End-to-end solution from input audio to transformed output

## 📋 Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- PyTorch
- Hugging Face account (for accessing models)

## 🚀 Installation
1. Clone the repository:
```bash
git clone https://github.com/okaditya84/Custom-AI-Voice-Modulation.git
cd Custom-AI-Voice-Modulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies:
```bash
pip install torch torchaudio
pip install pydub wave soundfile
pip install optimum
```

4. Login to Hugging Face:
```bash
huggingface-cli login
```

## 💻 Usage
The main functionality is provided in the `pipeline_voice_modulation.ipynb` notebook, which can be run in Google Colab or locally.

### Basic Usage:
```python
import os
from pydub import AudioSegment

# Run the voice modulation pipeline on an audio file
input_file = "path/to/input_audio.mp3"
output_file = "transformed_output.wav"

# Process the audio through the pipeline
main_pipeline(input_file)
```

### Advanced Usage:
You can customize the persona used for responses by modifying the system prompt in the `generate_response()` function.

## 🔧 Pipeline Components
- Whisper Model: Transcribes audio to text with high accuracy
- Text Generation Model: Transforms the transcribed text into responses based on defined personas
- TTS Model: Converts the generated text back to speech
- Audio Processing: Handles format conversion and combining audio segments

## ⚠️ System Requirements
- Memory: At least 8GB RAM, 16GB recommended
- GPU: CUDA-compatible GPU with 8GB+ VRAM for optimal performance
- Storage: Approximately 5GB for model downloads

## 🔍 Troubleshooting
- CUDA Out of Memory Errors: Try setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as an environment variable
- Audio Quality Issues: Ensure input audio is clear and has minimal background noise
- Model Download Failures: Check your internet connection and Hugging Face authentication

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments
- Hugging Face for providing access to state-of-the-art models
- OpenAI for the Whisper ASR system
- PyTorch team for their excellent deep learning framework