from pathlib import Path
from app.core.llm_client import LLMClient

def process_transcript(transcript_path: Path) -> str:
    """
    Reads a transcript file and uses LLM to extract a concise, structured
    summary of the key knowledge points to be added to the knowledge base.
    """
    text = transcript_path.read_text(encoding="utf-8")

    # We use LLM to condense the raw transcript (which often contains noise,
    # filler words, and conversational tangents) into factual knowledge.
    prompt = (
        "You are an expert educational content analyst. I will provide you with a "
        "class transcript. Your task is to extract the core factual knowledge, "
        "definitions, and key concepts discussed. \n\n"
        "Guidelines:\n"
        "1. Remove all filler words, conversational noise, and administrative talk.\n"
        "2. Organize the information logically by topic.\n"
        "3. Use clear headings and bullet points.\n"
        "4. Ensure technical terms are preserved and explained if the teacher does so.\n\n"
        "Transcript:\n"
        f"{text}\n\n"
        "Processed Knowledge Output:"
    )

    try:
        llm = LLMClient()
        summary = llm.generate_text(prompt)
        return summary
    except Exception as e:
        print(f"Error processing transcript with LLM: {e}")
        # Fallback to a basic cleaned version of the text if LLM fails
        return f"Raw Transcript Content (LLM Processing failed):\n\n{text}"

def append_transcript_to_knowledge(knowledge_path: Path, transcript_text: str):
    """
    Appends the processed transcript knowledge to the knowledge.md file.
    """
    with open(knowledge_path, "a", encoding="utf-8") as f:
        f.write("\n\n## Transcript-Derived Knowledge\n")
        f.write(transcript_text)
