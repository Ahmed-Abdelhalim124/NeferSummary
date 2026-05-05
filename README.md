#  NeferSummary 

An AI call center analysis tool that automatically transcribes, analyzes, and summarizes customer service calls in any language.

##  Features

- **Automatic Speaker Diarization**: Separates agent and customer speech
- **Multi-language Support**: Processes calls in Arabic, English, Spanish, and 100+ languages
- **AI-Powered Transcription**: Uses Google Gemini for accurate transcription
- **Intelligent Summarization**: Generates concise narrative summaries in the original language
- **Interactive AI Assistant**: Get help improving summaries with chat interface
- **Editable Output**: Manually edit summaries with undo functionality
- **Export to JSON**: Save results for record-keeping

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Google Gemini API key
- Hugging Face token with pyannote.audio access

### Installation


```

 Configure API keys in the code:
   - Add your Google Gemini API key to `GEMINI_API_KEY`
   - Add your Hugging Face token to `PYANNOTE_TOKEN`

### Accept pyannote.audio Terms

Before running, accept the user conditions for pyannote models:
1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Click "Agree and access repository"

```

##  How to Use

1. **Upload Audio**: Click "Upload Call Recording" and select your audio file (MP3, WAV, M4A, etc.)
2. **Analyze**: Click " Analyze Call" to start processing
3. **Review**: Check the generated summary and transcripts
4. **Improve**: Use the AI assistant to refine the summary
5. **Export**: Download results as JSON for your records


##  Supported Languages

Works with any language including:
- Arabic (العربية)
- English
- Spanish (Español)
- French (Français)
- German (Deutsch)
- And 90 languages


##  Output Format

The system generates:
- **Summary**: Narrative paragraph (80-100 words) in original language
- **Agent Transcript**: Complete transcription of agent speech
- **Customer Transcript**: Complete transcription of customer speech
- **JSON Export**: All data in structured format with timestamp

