# BroadAxis MCP Server
from mcp.server.fastmcp import FastMCP
import json
import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from tavily import TavilyClient
from dotenv import load_dotenv

# File generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import uuid
import datetime
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ['PINECONE_API_KEY', 'TAVILY_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
    sys.exit(1)

# Connection to Pinecone
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("sample3")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR = os.path.join(BASE_DIR, "generated_files")

# Ensure generated files directory exists
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

# Utility functions
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    # Remove path separators and other potentially dangerous characters
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Ensure filename is not empty
    return sanitized if sanitized else 'unnamed'

# Create MCP server
mcp = FastMCP("Broadaxis-Server")


@mcp.tool()
def Broadaxis_knowledge_search(query: str):
    """
    Retrieves the most relevant company's (Broadaxis) information from the internal knowledge base in response to a company related query.
    This tool performs semantic search over a RAG-powered database containing details about Broadaxis's background, team, projects, responsibilities, and domain expertise. It is designed to support tasks such as retrieving the knowledge regarding the company, surfacing domain-specific experience.

    Args:
        query: A natural language request related to the company’s past work, expertise, or capabilities (e.g., "What are the team's responsibilities?").
    """
    try:
        query_embedding = embedder.embed_documents([query])[0]
        query_result = index.query(
            vector=[query_embedding],
            top_k=5,
            include_metadata=True,
            namespace=""
        )
        documents = [result['metadata']['text'] for result in query_result['matches']]
        return documents
    except Exception as e:
        return json.dumps({"error": f"Knowledge search failed: {str(e)}"})    


tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]) 

@mcp.tool()
def web_search_tool(query: str):
    """
    Performs a real-time web search using Tavily and returns relevant results
    (including title, URL, and snippet).

    Args:
        query: A natural language search query.

    Returns:
        A JSON string with the top search results.
    """
    try:
        # Perform the search
        results = tavily.search(query=query, search_depth="advanced", include_answer=False)

        # Extract and format results
        formatted = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content")
            }
            for r in results.get("results", [])
        ]

        return json.dumps({"results": formatted})

    except Exception as e:
        return json.dumps({"error": str(e)})
    
@mcp.tool()
def generate_pdf_document(title: str, content: str, filename: str = None) -> str:
    """
    Generate a PDF document with the provided title and content.

    Args:
        title: The title of the document
        content: The main content of the document (supports markdown formatting)
        filename: Optional custom filename (without extension)

    Returns:
        JSON string with file information including download path
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"document_{uuid.uuid4().hex[:8]}"

        # Sanitize and ensure filename doesn't have extension
        filename = sanitize_filename(filename.replace('.pdf', ''))

        # Create full file path
        file_path = os.path.join(GENERATED_FILES_DIR, f"{filename}.pdf")

        # Create PDF document
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        # Process content (convert markdown to HTML-like formatting for reportlab)
        content_lines = content.split('\n')
        for line in content_lines:
            if line.strip():
                # Handle basic markdown formatting
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['Heading1']))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading2']))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['Heading3']))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
                else:
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))

        # Build PDF
        doc.build(story)

        # Get file info
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "status": "success",
            "filename": f"{filename}.pdf",
            "file_path": file_path,
            "file_size": file_size,
            "download_url": f"/download/{filename}.pdf",
            "created_at": datetime.datetime.now().isoformat(),
            "type": "pdf"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def generate_word_document(title: str, content: str, filename: str = None) -> str:
    """
    Generate a Word document with the provided title and content.

    Args:
        title: The title of the document
        content: The main content of the document (supports basic markdown formatting)
        filename: Optional custom filename (without extension)

    Returns:
        JSON string with file information including download path
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"document_{uuid.uuid4().hex[:8]}"

        # Sanitize and ensure filename doesn't have extension
        filename = sanitize_filename(filename.replace('.docx', '').replace('.doc', ''))

        # Create full file path
        file_path = os.path.join(GENERATED_FILES_DIR, f"{filename}.docx")

        # Create Word document
        doc = Document()

        # Add title
        title_paragraph = doc.add_heading(title, level=1)
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add some space after title
        doc.add_paragraph()

        # Process content (handle basic markdown formatting)
        content_lines = content.split('\n')
        for line in content_lines:
            line = line.strip()
            if line:
                if line.startswith('# '):
                    doc.add_heading(line[2:], level=1)
                elif line.startswith('## '):
                    doc.add_heading(line[3:], level=2)
                elif line.startswith('### '):
                    doc.add_heading(line[4:], level=3)
                elif line.startswith('- ') or line.startswith('* '):
                    # Add bullet point
                    doc.add_paragraph(line[2:], style='List Bullet')
                elif any(line.startswith(f'{i}. ') for i in range(1, 100)):
                    # Add numbered list (supports 1-99)
                    # Find the position after the number and dot
                    dot_pos = line.find('. ')
                    if dot_pos != -1:
                        doc.add_paragraph(line[dot_pos + 2:], style='List Number')
                else:
                    # Regular paragraph
                    doc.add_paragraph(line)
            else:
                # Add empty paragraph for spacing
                doc.add_paragraph()

        # Save document
        doc.save(file_path)

        # Get file info
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "status": "success",
            "filename": f"{filename}.docx",
            "file_path": file_path,
            "file_size": file_size,
            "download_url": f"/download/{filename}.docx",
            "created_at": datetime.datetime.now().isoformat(),
            "type": "docx"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def generate_text_file(content: str, filename: str = None, file_extension: str = "txt") -> str:
    """
    Generate a text file with the provided content.

    Args:
        content: The content to write to the file
        filename: Optional custom filename (without extension)
        file_extension: File extension (txt, md, csv, json, etc.)

    Returns:
        JSON string with file information including download path
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"file_{uuid.uuid4().hex[:8]}"

        # Sanitize filename and ensure no extension
        filename = sanitize_filename(filename.split('.')[0])

        # Create full file path
        file_path = os.path.join(GENERATED_FILES_DIR, f"{filename}.{file_extension}")

        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Get file info
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "status": "success",
            "filename": f"{filename}.{file_extension}",
            "file_path": file_path,
            "file_size": file_size,
            "download_url": f"/download/{filename}.{file_extension}",
            "created_at": datetime.datetime.now().isoformat(),
            "type": file_extension
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def list_generated_files() -> str:
    """
    List all generated files available for download.

    Returns:
        JSON string with list of available files
    """
    try:
        files = []
        if os.path.exists(GENERATED_FILES_DIR):
            for filename in os.listdir(GENERATED_FILES_DIR):
                file_path = os.path.join(GENERATED_FILES_DIR, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

                    files.append({
                        "filename": filename,
                        "file_size": file_size,
                        "download_url": f"/download/{filename}",
                        "modified_at": file_modified.isoformat(),
                        "type": filename.split('.')[-1] if '.' in filename else "unknown"
                    })

        return json.dumps({
            "status": "success",
            "files": files,
            "count": len(files)
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })

@mcp.prompt(title="Summarize Uploaded Document(s)")
def summarize_documents():
    """Summarize the content of one or more uploaded documents."""
    return f"""You are BroadAxis-AI, an assistant trained to extract key insights from RFP, RFQ, and RFI documents.
    Please review the uploaded document and generate a professional structured summary with the following sections:
    first let me know what the document is about, then provide the following sections:
    ### Summary Output Format:
1. **Executive Summary**
2. **Key Requirements (bulleted)**
3. **Project Scope & Timeline**
4. **Budget Information** (if available)
5. **Evaluation Criteria**
6. **Submission Deadline**
7. **Key Risks & Opportunities**
8. **Contact Information**
Use only the actual text provided in each document. Do not hallucinate or invent content. If a section is not present, say “Not specified in the document.”
At the end of all summaries, ask the user:
📊 **Would you like to know any additional information? Would you like my recommendation for a Go/No-Go decision on this opportunity?**
"""

@mcp.prompt(title="Go/No-Go Recommendation")
def go_no_go_recommendation() -> str:
    return """
You are BroadAxis-AI, an assistant trained to evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity.
The user has uploaded one or more opportunity documents. You have already summarized them.
Now perform a structured **Go/No-Go analysis** using the following steps:
---
### 🧠 Step-by-Step Evaluation Framework

1. **Review the RFP Requirements**
   - Highlight the most critical needs and evaluation criteria.

2. **Search Internal Knowledge** (via Broadaxis_knowledge_search)
   - Identify relevant past projects
   - Retrieve proof of experience in similar domains
   - Surface known strengths or capability gaps

3. **Evaluate Capability Alignment**
   - Estimate percentage match (e.g., "BroadAxis meets ~85% of the requirements")
   - Note any missing capabilities or unclear requirements

4. **Assess Resource Requirements**
   - Are there any specialized skills, timelines, or staffing needs?
   - Does BroadAxis have the necessary team or partners?

5. **Evaluate Competitive Positioning**
   - Based on known experience and domain, would BroadAxis be competitive?

Use only verified internal information (via Broadaxis_knowledge_search) and the uploaded documents.
Do not guess or hallucinate capabilities. If information is missing, clearly state what else is needed for a confident decision.

"""

@mcp.prompt(title="Generate Proposal or Capability Statement")
def generate_capability_statement() -> str:
    return """
You are BroadAxis-AI, an assistant trained to generate high-quality capability statements and proposal documents for RFP and RFQ responses.
The user has either uploaded an opportunity document or requested a formal proposal. Use all available information from:

- Uploaded documents (RFP/RFQ)
- Internal knowledge (via Broadaxis_knowledge_search)
- Prior summaries or analyses already provided

---

### 🧠 Instructions

- Do not invent names, projects, or facts.
- Use Broadaxis_knowledge_search to populate all relevant content.
- Leave placeholders where personal or business info is not available.
- Maintain professional, confident, and compliant tone.

If this proposal is meant to be saved, offer to generate a PDF or Word version using the appropriate tool.

"""

@mcp.prompt(title="Fill in Missing Information")
def fill_missing_information() -> str:
    return """
You are BroadAxis-AI, an intelligent assistant designed to fill in missing fields, answer RFP/RFQ questions, and complete response templates **strictly using verified information**.
The user has uploaded a document (PDF, DOCX, form, or Q&A table) that contains blank fields, placeholders, or questions. Your task is to **complete the missing sections** with reliable information from:

1. ✅ Broadaxis_knowledge_search (internal database)
2. ✅ The uploaded document itself
3. ✅ Prior chat context (if available)
---
### 🧠 RULES (Strict Compliance)

- ❌ **DO NOT invent or hallucinate** company details, financials, certifications, team names, or client info.
- ❌ **DO NOT guess** values you cannot verify.
- 🔐 If the question involves **personal, legal, or confidential information**, **do not fill it**.
- ✅ Use internal knowledge only when it clearly answers the field.
---
### ✅ Final Instruction
Only fill what you can verify using Broadaxis_knowledge_search and uploaded content. Leave everything else with clear, professional placeholders.

"""

@mcp.prompt(title="Configure Word Document Formatting")
def configure_word_formatting(
    font: str = "Calibri",
    font_size: int = 11,
    heading_font_size: int = 14,
    heading_bold: bool = True,
    line_spacing: float = 1.15,
    margin_top: str = "1 inch",
    margin_bottom: str = "1 inch",
    margin_left: str = "1 inch",
    margin_right: str = "1 inch",
    page_orientation: str = "portrait"
) -> str:
    """
    Prompt to configure Word document formatting settings.
    Users can override any of the default values.
    """
    return f"""Please use the following Word document formatting settings:

### 📝 Formatting Instructions

- **Font**: {font}
- **Font Size**: {font_size} pt
- **Heading Font Size**: {heading_font_size} pt
- **Heading Bold**: {"Yes" if heading_bold else "No"}
- **Line Spacing**: {line_spacing}
- **Margins**:
  - Top: {margin_top}
  - Bottom: {margin_bottom}
  - Left: {margin_left}
  - Right: {margin_right}
- **Page Orientation**: {page_orientation.capitalize()}

Apply these settings when generating Word documents, including proposals, summaries, and responses. Ensure consistency and professionalism throughout the layout.
"""


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    mcp.run(transport="streamable-http")
    