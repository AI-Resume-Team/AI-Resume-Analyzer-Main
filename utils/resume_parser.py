import re
import io

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file stream using PyPDF2."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


def extract_text_from_docx(file_stream):
    """Extract text from a DOCX file stream using python-docx."""
    try:
        import docx
        document = docx.Document(file_stream)
        text = "\n".join([para.text for para in document.paragraphs if para.text.strip()])
        return text.strip()
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"


def parse_resume(file_stream, filename):
    """
    Detect file type by extension and extract text accordingly.
    Returns extracted text as a string.
    """
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_stream)
    elif filename_lower.endswith(".docx"):
        return extract_text_from_docx(file_stream)
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."


def extract_email(text):
    """Extract the first email address found in the text."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else "Not found"


def extract_phone(text):
    """Extract the first phone number found in the text."""
    pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else "Not found"


def extract_name(text):
    """
    Heuristic: assume the candidate's name is on the first non-empty line
    that doesn't look like an email, phone, or URL.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:
        if "@" in line or re.search(r"\d{5,}", line) or "http" in line.lower():
            continue
        # A name typically has 2-4 words and no special characters
        if 1 < len(line.split()) <= 5 and re.match(r"^[A-Za-z\s.\-']+$", line):
            return line
    return "Not found"


def detect_sections(text):
    """
    Detect which standard resume sections are present.
    Returns a dict of section_name -> bool.
    """
    section_keywords = {
        "Education": ["education", "academic", "qualification", "degree", "university", "college", "school"],
        "Experience": ["experience", "employment", "work history", "internship", "job", "professional"],
        "Skills": ["skills", "technical skills", "technologies", "competencies", "expertise"],
        "Projects": ["projects", "personal projects", "academic projects", "portfolio"],
        "Certifications": ["certification", "certificate", "certified", "course", "training"],
        "Summary": ["summary", "objective", "profile", "about me", "career objective"],
    }
    text_lower = text.lower()
    found = {}
    for section, keywords in section_keywords.items():
        found[section] = any(kw in text_lower for kw in keywords)
    return found
