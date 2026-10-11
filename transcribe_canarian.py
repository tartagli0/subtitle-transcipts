import datetime
from faster_whisper import WhisperModel

def format_srt_time(seconds: float) -> str:
    """Converts seconds into standard SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def transcribe_movie_max_accuracy_cpu(audio_path: str, output_srt_path: str):
    print("Loading FULL 32-layer large-v3 model...")
    
    # 1. 'large-v3' provides the most robust decoder for regional dialect reconstruction
    # 2. 'int8_float16' keeps weights low for CPU but ups math execution resolution
    # 3. cpu_threads=4 assigns your physical cores without choking your operating system
    model = WhisperModel(
        "large-v3", 
        device="cpu", 
        compute_type="int8",
        cpu_threads=4 
    )

    # Dialect priming prompt for the Canarian accent mechanics
    canarian_prompt = (
        "Transcripción de una película rodada en Canarias. Chacho, vamos en la guagua. "
        "Habla canaria con seseo y aspiración. Transcribir fielmente el castellano."
    )

    print("Processing audio track with MAX accuracy parameters. This will take several hours...")
    segments, info = model.transcribe(
        audio_path,
        language="es",                  # Forcing Spanish forces Amazigh to fail confidence checks
        initial_prompt=canarian_prompt,   # Accent bias
        beam_size=10,                   # MAX deep-search for sentence pathways (heavy CPU load)
        patience=1.5,                   # Forces the beam search to look significantly longer for correct tracks
        best_of=5,                      # Generates 5 alternate text variations internally and picks the best one
        temperature=0.0,                # Strict, deterministic decoding logic
        vad_filter=True,                # Crucial to strip music/explosions so the CPU doesn't waste hours on noise
        vad_parameters=dict(min_speech_duration_ms=250, threshold=0.5)
    )

    print(f"Writing subtitles to: {output_srt_path}")
    
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        subtitle_index = 1
        
        for segment in segments:
            # High-precision confidence filter. 
            # Amazigh speech forced through the full Spanish model will fail miserably,
            # triggering this filter and cleanly keeping it out of your SRT file.
            if segment.no_speech_prob > 0.40 or getattr(segment, 'avg_logprob', 0) < -1.0:
                continue
                
            start_time = format_srt_time(segment.start)
            end_time = format_srt_time(segment.end)
            text = segment.text.strip()
            
            if not text:
                continue

            srt_file.write(f"{subtitle_index}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")
            
            subtitle_index += 1

    print(f"Successfully generated {subtitle_index - 1} high-accuracy Spanish subtitle blocks!")

if __name__ == "__main__":
    INPUT_AUDIO = "La_Piel_del_Volcan_2021.aac" 
    OUTPUT_SRT = "movie_canarian_high_accuracy.srt"
    
    transcribe_movie_max_accuracy_cpu(INPUT_AUDIO, OUTPUT_SRT)
