import os
import io
import re
import sys
import time
from typing import List, Tuple, Optional

import streamlit as st

# Text extraction
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from pptx import Presentation

# Summarization (TextRank via sumy)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# NLP helpers
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

# Optional: OpenAI for quiz generation
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Ensure NLTK data


def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")


ensure_nltk()

# -----------------------------
# Utility: Text extraction
# -----------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        text = pdf_extract_text(f)
    return text or ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = DocxDocument(f)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_pptx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        prs = Presentation(f)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                texts.append(shape.text)
    return "\n".join(texts)


def clean_text(text: str) -> str:
    # Basic cleanup: normalize whitespace, remove repeated spaces
    text = re.sub(r"\s+", " ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text.strip()

# -----------------------------
# Summarization (TextRank)
# -----------------------------


def summarize_textrank(text: str, sentence_count: int = 8) -> str:
    if not text or len(text.split()) < 50:
        return text  # Not enough content to summarize meaningfully
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    # Limit sentence_count to avoid empty summaries
    sentence_count = max(
        3, min(sentence_count, len(parser.document.sentences)))
    summary_sentences = summarizer(parser.document, sentence_count)
    summary = " ".join([str(s) for s in summary_sentences])
    return summary

# -----------------------------
# Rule-based quiz generator (fallback)
# -----------------------------


def generate_quiz_fallback(summary: str, num_questions: int = 5) -> List[dict]:
    """
    Create simple MCQs from the summary:
    - Extract key nouns/noun phrases
    - Build cloze-style questions
    - Generate distractors from other nouns
    """
    if not summary:
        return []

    sentences = sent_tokenize(summary)
    tokens = word_tokenize(summary)
    tagged = pos_tag(tokens)
    nouns = [w for w, t in tagged if t.startswith("NN") and w.isalpha()]
    nouns = [n for n in nouns if n.lower() not in stopwords.words("english")]
    nouns_unique = list(dict.fromkeys(nouns))  # preserve order, deduplicate

    # If too few nouns, fall back to sentence-based questions
    if len(nouns_unique) < 4:
        return sentence_gap_questions(sentences, num_questions)

    distractor_pool = nouns_unique[:]
    questions = []
    i = 0
    for s in sentences:
        # pick a noun in the sentence
        s_tokens = word_tokenize(s)
        s_tags = pos_tag(s_tokens)
        s_nouns = [w for w, t in s_tags if t.startswith("NN") and w.isalpha()]
        s_nouns = [n for n in s_nouns if n.lower(
        ) not in stopwords.words("english")]
        if not s_nouns:
            continue
        answer = s_nouns[0]
        # Build stem by replacing answer with blank
        stem = re.sub(rf"\b{re.escape(answer)}\b",
                      "_____", s, flags=re.IGNORECASE)

        # Distractors: pick 3 different nouns
        distractors = []
        for d in distractor_pool:
            if d.lower() != answer.lower() and d not in distractors:
                distractors.append(d)
            if len(distractors) == 3:
                break
        options = [answer] + distractors
        # Shuffle deterministically by index to keep reproducible without random
        options = rotate_list(options, i)

        questions.append({
            "stem": stem,
            "options": options,
            "answer": answer
        })
        i += 1
        if len(questions) >= num_questions:
            break

    if len(questions) < num_questions:
        # Top up with sentence gap questions
        extra = sentence_gap_questions(
            sentences, num_questions - len(questions))
        questions.extend(extra)

    return questions


def sentence_gap_questions(sentences: List[str], count: int) -> List[dict]:
    qs = []
    i = 0
    for s in sentences:
        words = [w for w in word_tokenize(s) if w.isalpha()]
        if len(words) < 6:
            continue
        # pick a mid word as answer
        idx = len(words) // 2
        answer = words[idx]
        stem = s.replace(answer, "_____")
        # naive distractors: nearby words
        distractors = []
        for w in words[:idx] + words[idx+1:]:
            if w.lower() != answer.lower() and w not in distractors:
                distractors.append(w)
            if len(distractors) == 3:
                break
        options = [answer] + distractors
        options = rotate_list(options, i)
        qs.append({"stem": stem, "options": options, "answer": answer})
        i += 1
        if len(qs) >= count:
            break
    return qs


def rotate_list(lst: List[str], k: int) -> List[str]:
    if not lst:
        return lst
    k = k % len(lst)
    return lst[k:] + lst[:k]

# -----------------------------
# OpenAI-based quiz generator
# -----------------------------


def generate_quiz_openai(summary: str, num_questions: int = 5) -> Optional[List[dict]]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not OPENAI_AVAILABLE or not api_key:
        return None

    openai.api_key = api_key
    prompt = f"""
You are a helpful assistant that creates high-quality multiple-choice quizzes.
Create {num_questions} MCQs from the following summary. Each question should have:
- A clear stem
- Exactly 4 options (A-D)
- One correct answer
- Provide the correct option letter

Return JSON with a list under "questions", where each item has:
- "stem": string
- "options": [A, B, C, D]
- "answer": "A"|"B"|"C"|"D"

Summary:
\"\"\"{summary}\"\"\""""

    try:
        # Use Responses API if available; otherwise fall back to Chat Completions
        # Chat Completions (widely supported)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You create structured MCQ quizzes."},
                {"role": "user", "content": prompt}
            ]
        )
        content = completion["choices"][0]["message"]["content"]
        # Try to parse JSON from the response
        import json
        data = json.loads(extract_json_block(content))
        questions = []
        for q in data.get("questions", []):
            stem = q.get("stem", "").strip()
            options = q.get("options", [])
            answer_letter = q.get("answer", "").strip()
            # Map letter to actual answer text
            letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            answer_text = options[letter_map.get(
                answer_letter, 0)] if options else ""
            questions.append({
                "stem": stem,
                "options": options,
                "answer": answer_text
            })
        return questions or None
    except Exception:
        return None


def extract_json_block(text: str) -> str:
    """
    Extract the first JSON-like block from text.
    """
    import re
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return '{"questions": []}'


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DocSummarizer + Quiz (TextRank)",
                   page_icon="🧠", layout="wide")


def app_header():
    st.title("🧠 Document Summarizer + AI Quiz")
    st.markdown(
        "Upload a PDF, Word (.docx), or PowerPoint (.pptx). "
        "The app extracts text, summarizes it using TextRank, and generates a quiz. "
        "If `OPENAI_API_KEY` is set, the quiz uses an LLM; otherwise, a smart fallback is used."
    )


@st.cache_data(show_spinner=False)
def cached_summarize(text: str, sentence_count: int) -> str:
    return summarize_textrank(text, sentence_count)


@st.cache_data(show_spinner=False)
def cached_quiz(summary: str, num_questions: int, use_llm: bool) -> List[dict]:
    if use_llm:
        q = generate_quiz_openai(summary, num_questions)
        if q:
            return q
    return generate_quiz_fallback(summary, num_questions)


def main():
    app_header()

    with st.sidebar:
        st.header("Settings")
        sentence_count = st.slider("Summary length (sentences)", 3, 15, 8)
        num_questions = st.slider("Quiz questions", 3, 15, 6)
        use_llm = st.toggle(
            "Use OpenAI (requires OPENAI_API_KEY)", value=False)
        st.caption(
            "Tip: Set OPENAI_API_KEY in your environment to enable LLM quiz generation.")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=False
    )

    if uploaded is None:
        st.info("Upload a file to begin.")
        return

    # Extract text based on type
    file_bytes = uploaded.read()
    ext = os.path.splitext(uploaded.name)[-1].lower()

    with st.spinner("Extracting text..."):
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(file_bytes)
        elif ext == ".docx":
            raw_text = extract_text_from_docx(file_bytes)
        elif ext == ".pptx":
            raw_text = extract_text_from_pptx(file_bytes)
        else:
            st.error("Unsupported file type.")
            return

    raw_text = clean_text(raw_text)

    if not raw_text or len(raw_text.split()) < 30:
        st.warning("Not enough text found in the document to summarize.")
        st.text_area("Extracted text (preview)", raw_text, height=200)
        return

    # Summarize
    with st.spinner("Summarizing with TextRank..."):
        summary = cached_summarize(raw_text, sentence_count)

    # Display results
    st.subheader("Summary")
    st.write(summary)

    # Quiz
    with st.spinner("Generating quiz..."):
        questions = cached_quiz(summary, num_questions, use_llm)

    st.subheader("Quiz")
    if not questions:
        st.warning("Could not generate quiz.")
    else:
        score = 0
        user_answers = []
        for idx, q in enumerate(questions, start=1):
            st.markdown(f"**Q{idx}.** {q['stem']}")
            options = q["options"]
            # Ensure 4 options for consistent UI
            if len(options) < 4:
                # pad with blanks
                options = options + ["(none)"] * (4 - len(options))
            choice = st.radio(
                f"Select answer for Q{idx}",
                options,
                index=0,
                key=f"q_{idx}"
            )
            user_answers.append((choice, q["answer"]))
            st.divider()

        if st.button("Submit Quiz"):
            for i, (chosen, correct) in enumerate(user_answers, start=1):
                is_correct = str(chosen).strip().lower() == str(
                    correct).strip().lower()
                if is_correct:
                    score += 1
                st.write(
                    f"Q{i}: {'✅ Correct' if is_correct else '❌ Incorrect'} — Your answer: {chosen} | Correct: {correct}")
            st.success(f"Your score: {score}/{len(user_answers)}")

    # Optional: show extracted text preview
    with st.expander("Show extracted text"):
        st.text_area("Document text", raw_text, height=250)

    st.caption(
        "Built with Streamlit, Sumy (TextRank), and optional OpenAI for quiz generation.")


if __name__ == "__main__":
    main()
