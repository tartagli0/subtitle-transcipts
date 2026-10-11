from faster_whisper import WhisperModel

audio_file = "La_Piel_del_Volcan_2021.aac"
output_srt = "La_Piel_del_Volcan_2021.srt"

print("Loading Whisper model optimized for CPU...")
model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)

print("Transcribing Spanish audio directly from AAC...")
segments, info = model.transcribe(audio_file, beam_size=4, language="es")

def format_time(seconds):
    """Formats raw seconds into standard SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

print("Writing subtitles to SRT format...")
with open(output_srt, "w", encoding="utf-8") as f:
    for idx, segment in enumerate(segments, start=1):
        start = format_time(segment.start)
        end = format_time(segment.end)
        text = segment.text.strip()
        
        f.write(f"{idx}\n")
        f.write(f"{start} --> {end}\n")
        f.write(f"{text}\n\n")

print(f"Done! Subtitles saved to: {output_srt}")
