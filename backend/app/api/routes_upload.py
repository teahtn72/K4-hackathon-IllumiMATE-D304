from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from pathlib import Path
import json
import re
import uuid
from typing import List
from app.core.config import DATA_PACK_DIR
from app.pipeline.pdf_processor import extract_slide_ranges, update_knowledge_md, extract_pages_content
from app.core.llm_client import LLMClient
from app.pipeline.transcript_processor import process_transcript, append_transcript_to_knowledge
from app.core.session_store import get_session, save_session, create_session
from app.core.schemas import TranscriptSegment, ClassifiedSegment, OutlineSection, Slide, AlignmentItem
from app.pipeline.outline import outline_json, parse_transcript
from app.pipeline.align import align_sections
from app.pipeline.classify import classify_with_llm
from app.pipeline.quiz_bank import generate_quiz, QuizGenerationError
from app.utils.pdf_extract import extract_pdf_pages, parse_slide_outline

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = DATA_PACK_DIR / "uploads"
KNOWLEDGE_FILE = DATA_PACK_DIR / "knowledge.md"

# Global state to track enrichment status. In production, use Redis or a database.
ENRICHMENT_STATUS = {
    "is_running": False,
    "last_updated": None
}

def _pdf_alignment(sections, transcript_path: Path = None):
    """
    Simplified alignment logic to match transcript segments to slide sections.
    """
    path = None
    if transcript_path and transcript_path.is_file():
        try:
            transcript_text = transcript_path.read_text(encoding="utf-8")
            transcript_sections = parse_transcript(transcript_text)
            if transcript_sections:
                path = transcript_path
        except Exception as e:
            print(f"Error reading/parsing uploaded transcript: {e}")

    if not path:
        from app.core.config import TRANSCRIPT_DIR
        # Use a default transcript for alignment if no specific one is provided
        transcript_file = "transcript-01-clean.md"
        path = TRANSCRIPT_DIR / transcript_file

    if not path.is_file():
        return [], [], []

    transcript_text = path.read_text(encoding="utf-8")
    transcript_sections = parse_transcript(transcript_text)

    all_segments = []
    for section in transcript_sections:
        all_segments.extend(section.segments)

    try:
        classified = classify_with_llm([
            {"segment_id": segment.segment_id, "text": segment.text}
            for segment in all_segments
        ])
    except Exception as exc:
        print(f"Alignment classification failed: {exc}")
        return all_segments, [], []

    alignment = align_sections(sections, all_segments)
    return all_segments, classified, alignment

def background_enrich_knowledge(weak_sections: list[str], pdf_filename: str):
    """
    Background task to scan PDF for text and visual content and enrich knowledge.md
    """
    global ENRICHMENT_STATUS
    ENRICHMENT_STATUS["is_running"] = True
    try:
        pdf_path = UPLOAD_DIR / pdf_filename
        if not pdf_path.exists():
            ENRICHMENT_STATUS["is_running"] = False
            return

        ranges = extract_slide_ranges(str(pdf_path))
        enrichments = []
        for section in weak_sections:
            matching_range = next((r for r in ranges if section.lower() in r["title"].lower()), None)
            if matching_range:
                content = extract_pages_content(str(pdf_path), matching_range["start_page"], matching_range["end_page"])
                prompt = (
                    f"You are an educational assistant. I will provide you with the text content of a few slides "
                    f"belonging to the section '{section}'.\n\n"
                    f"Content:\n{content}\n\n"
                    f"Based on this content and the typical structure of a presentation, please describe any "
                    f"likely images, diagrams, charts, or complex formulas that would be present in these slides "
                    f"to help a student understand the concepts better. Be specific about what the visual would show."
                )
                visual_insights = "No specific visual insights found."
                try:
                    llm = LLMClient()
                    visual_insights = llm.generate_text(prompt)
                except Exception as e:
                    print(f"AI enrichment failed for {section}: {e}")

                enrichments.append(
                    f"### Detailed Knowledge for {section}\n"
                    f"**Slide Text Content:**\n{content}\n\n"
                    f"**Visual/Formula Insights:**\n{visual_insights}\n"
                )

        if enrichments:
            with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
                f.write("\n\n## AI Enriched Knowledge (Weak Sections)\n")
                f.write("\n".join(enrichments))
    except Exception as e:
        print(f"Background enrichment failed: {e}")
    finally:
        ENRICHMENT_STATUS["is_running"] = False

@router.post("/package")
async def upload_package(files: List[UploadFile] = File(...)):
    """
    Uploads a PDF slide and a transcript, processes them,
    populates the knowledge base and session, and generates a quiz.
    """
    pdf_file = None
    transcript_file = None

    for f in files:
        if f.filename.lower().endswith(".pdf"):
            pdf_file = f
        elif f.filename.lower().endswith((".txt", ".vtt", ".srt", ".md")):
            transcript_file = f

    if not pdf_file or not transcript_file:
        raise HTTPException(status_code=400, detail="Both a PDF slide and a transcript file are required.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = UPLOAD_DIR / pdf_file.filename
    with pdf_path.open("wb") as buffer:
        buffer.write(await pdf_file.read())

    transcript_path = UPLOAD_DIR / transcript_file.filename
    with transcript_path.open("wb") as buffer:
        buffer.write(await transcript_file.read())

    try:
        # Process PDF
        pdf_bytes = pdf_path.read_bytes()
        sections = parse_slide_outline(pdf_bytes)
        if not sections:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF slides")

        ranges = extract_slide_ranges(str(pdf_path))
        update_knowledge_md(str(KNOWLEDGE_FILE), ranges)

        # Process Transcript
        processed_transcript_text = process_transcript(transcript_path)
        append_transcript_to_knowledge(KNOWLEDGE_FILE, processed_transcript_text)

        # Initialize Session
        session = create_session()
        session_id = session.session_id

        transcript_segments, classified, alignment = _pdf_alignment(sections, transcript_path=transcript_path)

        pages = extract_pdf_pages(pdf_bytes)
        slides = [
            Slide(
                slide_id=f"P{index:02d}", page_number=index,
                title=section.title,
                text=pages[index - 1] if index <= len(pages) else "",
                segment_ids=next((item["related_segment_ids"] for item in alignment if item["section_id"] == section.section_id), []),
            )
            for index, section in enumerate(sections, start=1)
        ]

        segment_slide = {
            segment_id: f"P{index:02d}"
            for index, item in enumerate(alignment, start=1)
            for segment_id in item["related_segment_ids"]
        }

        session.raw_transcript = [
            TranscriptSegment(
                segment_id=item.segment_id, text=item.text,
                slide_id=segment_slide.get(item.segment_id),
            )
            for item in transcript_segments
        ]
        session.classified_transcript = [
            ClassifiedSegment(
                segment_id=item["segment_id"],
                text=next((segment.text for segment in transcript_segments if segment.segment_id == item["segment_id"]), ""),
                label=item["label"],
            )
            for item in classified
        ]
        session.outline = [OutlineSection(**item) for item in outline_json(sections)]
        session.slides = slides
        session.alignment = [AlignmentItem(**item) for item in alignment]
        session.source = "slide_pdf"
        save_session(session)

        # Generate Quiz
        try:
            questions = generate_quiz(sections, n_questions=20)
        except QuizGenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        return {
            "id": session_id,
            "fileNames": [pdf_file.filename, transcript_file.filename],
            "sectionCount": len(session.outline),
            "quiz": questions
        }

    except Exception as e:
        print(f"Error processing package: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    try:
        ranges = extract_slide_ranges(str(file_path))
        update_knowledge_md(str(KNOWLEDGE_FILE), ranges)

        titles = [r["title"] for r in ranges]
        prompt = (
            "You are analyzing a lecture.\n\n"
            "Here are all slide titles in order:\n"
            f"{titles}\n\n"
            "Your task:\n"
            "1. Divide this lecture into major learning units.\n"
            "2. Assign each slide to one learning unit.\n"
            "3. Give each learning unit a concise name.\n"
            "4. Return JSON only. Ví dụ output: [\n"
            "  {\n"
            "    \"unit\": \"Transformer Overview\",\n"
            "    \"slides\": [1,2]\n"
            "  },\n"
            "  {\n"
            "    \"unit\": \"Encoder\",\n"
            "    \"slides\": [3,4,5]\n"
            "  },\n"
            "  {\n"
            "    \"unit\": \"Self Attention\",\n"
            "    \"slides\": [6,7,8,9]\n"
            "  }\n"
            "]"
        )

        llm = LLMClient()
        response_text = llm.generate_text(prompt)
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        learning_units = json.loads(json_match.group()) if json_match else []

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "ranges": ranges,
            "learning_units": learning_units,
            "knowledge_path": str(KNOWLEDGE_FILE)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/enrich")
async def enrich_knowledge(
    background_tasks: BackgroundTasks,
    payload: dict
):
    weak_sections = payload.get("weak_sections", [])
    pdf_filename = payload.get("pdf_filename", "")
    if not pdf_filename:
        raise HTTPException(status_code=400, detail="pdf_filename is required")
    background_tasks.add_task(background_enrich_knowledge, weak_sections, pdf_filename)
    return {"message": "Enrichment process started in background"}

@router.get("/enrich-status")
async def get_enrichment_status():
    return ENRICHMENT_STATUS
