"""
Broadaxis MCP Server v2

- Loads Pinecone and HuggingFace embeddings
- Exposes a semantic search tool over a Pinecone index
- Provides multi-step prompt templates for RFP/RFQ/RFI workflows

Prereqs (suggested):
    pip install python-dotenv langchain-huggingface pinecone-client mcp fastmcp

Required env vars:
    PINECONE_API_KEY
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from mcp.server.fastmcp import FastMCP


# -----------------------------
# Environment & Client Setup
# -----------------------------
load_dotenv()

REQUIRED_ENV_VARS = ["PINECONE_API_KEY"]
missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
if missing:
    print(
        f"Error: Missing required environment variables: {', '.join(missing)}",
        file=sys.stderr,
    )
    sys.exit(1)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Initialize Pinecone client & index
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "sample3"
index = pc.Index(INDEX_NAME)

# Initialize embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create MCP server
mcp = FastMCP("Broadaxis-Server-v2", host="0.0.0.0", port=8895)


# -----------------------------
# Tools
# -----------------------------
@mcp.tool()
def Broadaxis_knowledge_search(query: str):
    """
    Retrieves the most relevant Broadaxis information from the internal knowledge base
    in response to a company-related query.

    This tool performs semantic search over a RAG-powered database containing details
    about Broadaxis's background, team, projects, responsibilities, and domain expertise.

    Args:
        query: Natural language request related to the company’s past work, expertise, or capabilities.
               (e.g., "What are the team's responsibilities?")

    Returns:
        List[str]: Top matching document texts, or a JSON error string on failure.
    """
    try:
        # Prefer embed_query for a single query string
        query_embedding = embedder.embed_query(query)

        # Pinecone expects a 1D vector for 'vector'
        query_result = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="",  # set if you use namespacing
        )

        matches = query_result.get("matches", []) or []
        documents = []
        for m in matches:
            meta = m.get("metadata", {}) or {}
            # Adjust the key name below if your metadata uses a different field
            text = meta.get("text")
            if text:
                documents.append(text)

        return documents

    except Exception as e:
        return json.dumps({"error": f"Knowledge search failed: {str(e)}"})


# -----------------------------
# Prompt Templates
# -----------------------------
@mcp.prompt(title="Identifying the Documents")
def Step1_Identifying_documents():
    """
    Read PDFs from filesystem path, categorize them as RFP/RFI/RFQ-related,
    fillable forms, or non-fillable documents.
    """
    return (
        "read the files from the provided filesystems tool path using PDFFiller tool to "
        "categorize each uploaded PDF into the following groups:\n"
        "1. 📘 **Primary Documents** — PDFs that contain RFP, RFQ, or RFI content "
        "(e.g., project scope, requirements, evaluation criteria).\n"
        "2. 📝 **Fillable Forms** — PDFs with interactive fields intended for user input "
        "(e.g., pricing tables, response forms).\n"
        "3. 📄 **Non-Fillable Documents** — PDFs that are neither RFP-type nor interactive, "
        "such as attachments or informational appendices.\n"
        "---\n"
        "Once the classification is complete:\n"
        "📊 **Would you like to proceed to the next step and generate summaries for the relevant documents?** "
        "If yes, please upload the files and attach the summary prompt template."
    )


@mcp.prompt(title="Step-2: Executive Summary of Procurement Document")
def Step2_summarize_documents():
    """
    Generate a clear, high-value summary of uploaded RFP, RFQ, or RFI documents
    for executive decision-making.
    """
    return (
        "You are **BroadAxis-AI**, an intelligent assistant that analyzes procurement documents (RFP, RFQ, RFI) "
        "to help vendor teams quickly understand the opportunity and make informed pursuit decisions.\n"
        "When a user uploads one or more documents, do the following **for each document, one at a time**:\n"
        "---\n"
        "### 📄 Document: [Document Name]\n"
        "#### 🔹 What is This About?\n"
        "> A 3–5 sentence **plain-English overview** of the opportunity. Include:\n"
        "- Who issued it (organization)\n"
        "- What they need / are requesting\n"
        "- Why (the business problem or goal)\n"
        "- Type of response expected (proposal, quote, info)\n"
        "---\n"
        "#### 🧩 Key Opportunity Details\n"
        "List all of the following **if available** in the document:\n"
        "- **Submission Deadline:** [Date + Time]\n"
        "- **Project Start/End Dates:** [Dates or Duration]\n"
        "- **Estimated Value / Budget:** [If stated]\n"
        "- **Response Format:** (e.g., PDF proposal, online portal, pricing form, etc.)\n"
        "- **Delivery Location(s):** [City, Region, Remote, etc.]\n"
        "- **Eligibility Requirements:** (Certifications, licenses, location limits)\n"
        "- **Scope Summary:** (Bullet points or short paragraph outlining main tasks or deliverables)\n"
        "---\n"
        "#### 📊 Evaluation Criteria\n"
        "How will responses be scored or selected? Include weighting if provided (e.g., 40% price, 30% experience).\n"
        "---\n"
        "#### ⚠️ Notable Risks or Challenges\n"
        "Mention anything that could pose a red flag or require clarification (tight timeline, vague scope, "
        "legal constraints, strict eligibility).\n"
        "---\n"
        "#### 💡 Potential Opportunities or Differentiators\n"
        "Highlight anything that could give a competitive edge or present upsell/cross-sell opportunities "
        "(e.g., optional services, innovation clauses, incumbent fatigue).\n"
        "---\n"
        "#### 📞 Contact & Submission Info\n"
        "- **Primary Contact:** Name, title, email, phone (if listed)\n"
        "- **Submission Instructions:** Portal, email, physical, etc.\n"
        "---\n"
        "### 🤔 Ready for Action?\n"
        "> Would you like a strategic assessment or a **Go/No-Go recommendation** for this opportunity?\n"
        "⚠️ Only summarize what is clearly and explicitly stated. Never guess or infer."
    )


@mcp.prompt(title="Step-3 : Go/No-Go Recommendation")
def Step3_go_no_go_recommendation() -> str:
    return (
        "You are BroadAxis-AI, an assistant trained to evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity. "
        "The user has uploaded one or more opportunity documents. You have already summarized them/if not ask for the user to "
        "upload RFP/RFI/RF documents and generate summary. Now perform a structured **Go/No-Go analysis** using the following steps:\n"
        "---\n"
        "### 🧠 Step-by-Step Evaluation Framework\n"
        "1. **Review the RFP Requirements**\n"
        "- Highlight the most critical needs and evaluation criteria.\n"
        "2. **Search Internal Knowledge** (via Broadaxis_knowledge_search)\n"
        "- Identify relevant past projects\n"
        "- Retrieve proof of experience in similar domains\n"
        "- Surface known strengths or capability gaps\n"
        "3. **Evaluate Capability Alignment**\n"
        "- Estimate percentage match (e.g., \"BroadAxis meets ~85% of the requirements\")\n"
        "- Note any missing capabilities or unclear requirements\n"
        "4. **Assess Resource Requirements**\n"
        "- Are there any specialized skills, timelines, or staffing needs?\n"
        "- Does BroadAxis have the necessary team or partners?\n"
        "5. **Evaluate Competitive Positioning**\n"
        "- Based on known experience and domain, would BroadAxis be competitive?\n"
        "Use only verified internal information (via Broadaxis_knowledge_search) and the uploaded documents. Do not guess or hallucinate capabilities. "
        "If information is missing, clearly state what else is needed for a confident decision.\n"
        "if your recommendation is a Go, list down the things to the user of the tasks he need to complete to finish the submission of RFP/RFI/RFQ."
    )


@mcp.prompt(title="Step-4 : Generate Proposal or Capability Statement")
def Step4_generate_capability_statement() -> str:
    return (
        "You are BroadAxis-AI, an assistant trained to generate high-quality capability statements and proposal documents for RFP and RFQ responses. "
        "The user has either uploaded an opportunity document or requested a formal proposal.\n"
        "Use all available information from:\n"
        "- Uploaded documents (RFP/RFQ)\n"
        "- Internal knowledge (via Broadaxis_knowledge_search)\n"
        "- Prior summaries or analyses already provided\n"
        "---\n"
        "### 🧠 Instructions\n"
        "- Do not invent names, projects, or facts.\n"
        "- Use Broadaxis_knowledge_search to populate all relevant content.\n"
        "- Leave placeholders where personal or business info is not available.\n"
        "- Maintain professional, confident, and compliant tone.\n"
        "If this proposal is meant to be saved, offer to generate a PDF or Word version using the appropriate tool."
    )


@mcp.prompt(title="Step-5 : Fill in Missing Information")
def Step5_fill_missing_information() -> str:
    return (
        "You are BroadAxis-AI, an intelligent assistant designed to fill in missing fields using ppdf filler tool , answer RFP/RFQ questions, "
        "and complete response templates **strictly using verified information**. Your task is to **complete the missing sections** on the fillable "
        "documents which you have identified previously with reliable information from:\n"
        "1. Broadaxis_knowledge_search (internal database)\n"
        "2. The uploaded document itself\n"
        "3. Prior chat context (if available)\n"
        "---\n"
        "### 🧠 RULES (Strict Compliance)\n"
        "- ❌ **DO NOT invent or hallucinate** company details, financials, certifications, team names, or client info.\n"
        "- ❌ **DO NOT guess** values you cannot verify.\n"
        "- 🔐 If the question involves **personal, legal, or confidential information**, **do not fill it**.\n"
        "- ✅ Use internal knowledge only when it clearly answers the field.\n"
        "---\n"
        "### ✅ Final Instruction\n"
        "Only fill what you can verify using Broadaxis_knowledge_search and uploaded content. Leave everything else with clear, professional placeholders."
    )


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
    page_orientation: str = "portrait",
) -> str:
    """
    Prompt to configure Word document formatting settings. Users can override any of the default values.
    """
    return (
        "Please use the following Word document formatting settings:\n"
        "### 📝 Formatting Instructions\n"
        f"- **Font**: {font}\n"
        f"- **Font Size**: {font_size} pt\n"
        f"- **Heading Font Size**: {heading_font_size} pt\n"
        f"- **Heading Bold**: {'Yes' if heading_bold else 'No'}\n"
        f"- **Line Spacing**: {line_spacing}\n"
        "- **Margins**:\n"
        f"  - Top: {margin_top}\n"
        f"  - Bottom: {margin_bottom}\n"
        f"  - Left: {margin_left}\n"
        f"  - Right: {margin_right}\n"
        f"- **Page Orientation**: {page_orientation.capitalize()}\n"
        "Apply these settings when generating Word documents, including proposals, summaries, and responses. "
        "Ensure consistency and professionalism throughout the layout."
    )


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mcp.run(transport="sse")
