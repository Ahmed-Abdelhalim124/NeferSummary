import gradio as gr
import torch
import torchaudio
from pyannote.audio import Pipeline
import google.generativeai as genai
from pydub import AudioSegment
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
import warnings
import concurrent.futures

warnings.filterwarnings('ignore')



GEMINI_API_KEY = 'Input API'
genai.configure(api_key=GEMINI_API_KEY)

PYANNOTE_TOKEN = "Input Token"

print("Loading speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=PYANNOTE_TOKEN
)

if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("✓ Using CPU")



def process_audio_file(audio_path, progress=gr.Progress()):
    
    progress(0.1, desc="Loading audio file...")
    
    audio = AudioSegment.from_file(audio_path)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
        audio.export(wav_path, format='wav')
    
    progress(0.2, desc="Running speaker diarization...")
    
    waveform, sample_rate = torchaudio.load(wav_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=2
    )
    
    progress(0.4, desc="Analyzing speakers...")
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end,
            'duration': turn.end - turn.start
        })
    
    df = pd.DataFrame(segments)
    
    speaker_stats = df.groupby('speaker').agg({
        'duration': ['sum', 'count']
    })
    speaker_stats.columns = ['total_time', 'num_segments']
    
    sorted_speakers = speaker_stats.sort_values('total_time', ascending=False)
    agent_speaker = sorted_speakers.index[0]
    customer_speaker = sorted_speakers.index[1]
    
    role_mapping = {agent_speaker: 'Agent', customer_speaker: 'Customer'}
    df['role'] = df['speaker'].map(role_mapping)
    
    progress(0.5, desc="Extracting audio segments...")
    
    agent_audio = AudioSegment.empty()
    customer_audio = AudioSegment.empty()
    
    for _, row in df.iterrows():
        start_ms = int(row['start'] * 1000)
        end_ms = int(row['end'] * 1000)
        segment = audio[start_ms:end_ms]
        
        if row['role'] == 'Agent':
            agent_audio += segment
        else:
            customer_audio += segment
    
    with tempfile.NamedTemporaryFile(suffix='_agent.wav', delete=False) as tmp_agent:
        agent_path = tmp_agent.name
        agent_audio.export(agent_path, format='wav')
    
    with tempfile.NamedTemporaryFile(suffix='_customer.wav', delete=False) as tmp_customer:
        customer_path = tmp_customer.name
        customer_audio.export(customer_path, format='wav')
    
    progress(0.6, desc="Uploading to Gemini...")
    
    agent_file = genai.upload_file(path=agent_path)
    customer_file = genai.upload_file(path=customer_path)
    
    progress(0.7, desc="Transcribing (parallel processing)...")
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    transcribe_prompt = """
    Transcribe this audio completely. Provide only the transcription text.
    Maintain the original language of the speech.
    """
    
    def transcribe_audio(file, role):
        response = model.generate_content([transcribe_prompt, file])
        return role, response.text
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(transcribe_audio, agent_file, 'agent'),
            executor.submit(transcribe_audio, customer_file, 'customer')
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    transcripts = {role: text for role, text in results}
    agent_transcript = transcripts['agent']
    customer_transcript = transcripts['customer']
    
    progress(0.9, desc="Generating summary...")
    
    summary_prompt = f"""
    You are analyzing a customer service call. Below are the transcriptions:

    AGENT:
    {agent_transcript}

    CUSTOMER:
    {customer_transcript}

    Please provide a narrative summary of this call in ONE CONTINUOUS PARAGRAPH following this style:

    Write a flowing narrative that describes:
    - Who initiated the call and their role/department (not agent name)
    - The main purpose of the call
    - The steps taken during the interaction (in chronological order and summary)
    - Any technical challenges or issues encountered
    - The outcome and whether the issue was resolved
    - Customer implications

    Write in past tense, third person perspective. Use smooth transitions between ideas. 
    Keep it concise but comprehensive - aim for 80-100 words. 
    Do NOT use bullet points, numbered lists, or section headers. 
    Write as one flowing paragraph in the same language as the transcripts.
    It will be read by another agent - be sure to give all necessary notes.
    """
    
    summary_response = model.generate_content(summary_prompt)
    call_summary = summary_response.text
    
    progress(1.0, desc="Complete!")
    
    os.unlink(wav_path)
    os.unlink(agent_path)
    os.unlink(customer_path)
    
    return {
        'summary': call_summary,
        'agent_transcript': agent_transcript,
        'customer_transcript': customer_transcript
    }



def chat_assistant(message, chat_history, current_summary, agent_transcript, customer_transcript):
    
    if not message.strip():
        return chat_history, "", current_summary
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    context_prompt = f"""
    You are an AI assistant helping a call center agent improve their call summary.
    
    CURRENT SUMMARY:
    {current_summary}
    
    AGENT TRANSCRIPT:
    {agent_transcript[:2000]}...
    
    CUSTOMER TRANSCRIPT:
    {customer_transcript[:2000]}...
    
    The agent is asking: {message}
    
    Instructions:
    - If the agent asks you to rewrite, improve, or modify the summary, provide the COMPLETE improved summary
    - If the agent asks a question or wants suggestions, provide helpful advice without rewriting
    - Be concise and actionable
    - Maintain the same language as the original summary
    - Keep summaries to 80-100 words in one paragraph
    """
    
    response = model.generate_content(context_prompt)
    bot_message = response.text
    
    chat_history.append((message, bot_message))
    
    words = bot_message.split()
    if len(words) > 50 and '\n\n' not in bot_message[:100]:
        return chat_history, "", bot_message
    else:
        return chat_history, "", current_summary



summary_history = []

def save_to_history(summary):
    global summary_history
    if not summary_history or summary_history[-1] != summary:
        summary_history.append(summary)
        if len(summary_history) > 10:
            summary_history.pop(0)
    return summary

def undo_summary():
    global summary_history
    if len(summary_history) > 1:
        summary_history.pop()
        return summary_history[-1]
    elif len(summary_history) == 1:
        return summary_history[0]
    else:
        return "No history to undo"



def analyze_call(audio_file):
    
    if audio_file is None:
        return "Please upload an audio file", "", "", []
    
    try:
        result = process_audio_file(audio_file)
        
        global summary_history
        summary_history = []  
        save_to_history(result['summary'])
        
        return (
            result['summary'],
            result['agent_transcript'],
            result['customer_transcript'],
            []  
        )
    
    except Exception as e:
        return f"Error: {str(e)}", "", "", []



def export_json(summary, agent_transcript, customer_transcript):
    
    data = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "summary": summary,
        "transcripts": {
            "agent": agent_transcript,
            "customer": customer_transcript
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        return f.name



def create_interface():
    
    with gr.Blocks(title="NeferSummary", theme=gr.themes.Soft()) as demo:
        
        agent_transcript_state = gr.State("")
        customer_transcript_state = gr.State("")
        
        gr.Markdown("# 🎧 NeferSummary")
        gr.Markdown("### Upload a call recording to analyze, transcribe, and summarize")
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Call Recording"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Call", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                with gr.Row():
                    undo_btn = gr.Button("↩️ Undo", size="sm")
                    export_btn = gr.Button("💾 Export JSON", size="sm", variant="secondary")
                
                export_file = gr.File(label="Download JSON", visible=False)
            
            with gr.Column(scale=2):
                summary_output = gr.Textbox(
                    label="Call Summary",
                    lines=10,
                    placeholder="Summary will appear here after analysis...",
                    interactive=True
                )
                
                gr.Markdown("### 💬 AI Assistant - Get help improving your summary")
                
                chatbot = gr.Chatbot(
                    height=300,
                    label="Chat with AI Assistant",
                    show_label=False
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask me to improve the summary or answer questions...",
                        show_label=False,
                        scale=4
                    )
                    chat_send = gr.Button("Send", scale=1, variant="primary")
        
        gr.Markdown("""
        ---
        ### 📖 Quick Guide
        1. **Upload** your call recording (MP3, WAV, M4A, etc.)
        2. **Click** "Analyze Call" to process
        3. **Edit** the summary directly or ask the AI assistant for help
        4. **Use chat** to request improvements: "make it more professional", "add more detail about the issue", etc.
        5. **Undo** changes if needed
        6. **Export** final results as JSON
        
        🌍 **Multi-language support** - Automatically processes calls in any language
        """)
        
       
        analyze_btn.click(
            fn=analyze_call,
            inputs=[audio_input],
            outputs=[
                summary_output,
                agent_transcript_state,
                customer_transcript_state,
                chatbot
            ]
        )
        
        summary_output.change(
            fn=save_to_history,
            inputs=[summary_output],
            outputs=[]
        )
        
        undo_btn.click(
            fn=undo_summary,
            inputs=[],
            outputs=[summary_output]
        )
        
        def handle_chat(message, history, summary, agent_trans, customer_trans):
            new_history, cleared_input, updated_summary = chat_assistant(
                message, history, summary, agent_trans, customer_trans
            )
            save_to_history(updated_summary)
            return new_history, cleared_input, updated_summary
        
        chat_send.click(
            fn=handle_chat,
            inputs=[
                chat_input,
                chatbot,
                summary_output,
                agent_transcript_state,
                customer_transcript_state
            ],
            outputs=[chatbot, chat_input, summary_output]
        )
        
        chat_input.submit(
            fn=handle_chat,
            inputs=[
                chat_input,
                chatbot,
                summary_output,
                agent_transcript_state,
                customer_transcript_state
            ],
            outputs=[chatbot, chat_input, summary_output]
        )
        
        export_btn.click(
            fn=export_json,
            inputs=[summary_output, agent_transcript_state, customer_transcript_state],
            outputs=[export_file]
        ).then(
            lambda: gr.File(visible=True),
            outputs=[export_file]
        )
    
    return demo



if __name__ == "__main__":
    print("\n" + "="*60)
    print("CALL CENTER ANALYSIS SYSTEM")
    print("="*60)
    print("\nStarting application...")
    
    demo = create_interface()
    
    demo.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",  
        server_port=7860        
    )