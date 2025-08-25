from mcp.server.fastmcp import FastMCP
import json
import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ['PINECONE_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
    sys.exit(1)

# Connection to Pinecone
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("sample3")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create MCP server
mcp = FastMCP("Broadaxis-Server-v2" , host="0.0.0.0",  port = 8893)

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
 

@mcp.prompt(title="Identifying the Documents")
def Step1_Identifying_documents():
    """read PDFs from filesystem path, categorize them as RFP/RFI/RFQ-related, fillable forms, or non-fillable documents."""
    return f"""read the files from the provided filesystems tool path using PDFFiller tool to categorize each uploaded PDF into the following groups:

1. 📘 **Primary Documents** — PDFs that contain RFP, RFQ, or RFI content (e.g., project scope, requirements, evaluation criteria).
2. 📝 **Fillable Forms** — PDFs with interactive fields intended for user input (e.g., pricing tables, response forms).
3. 📄 **Non-Fillable Documents** — PDFs that are neither RFP-type nor interactive, such as attachments or informational appendices.
---
Once the classification is complete:

📊 **Would you like to proceed to the next step and generate summaries for the relevant documents?**  
If yes, please upload the files and attach the summary prompt template.
"""

@mcp.prompt(title="Step-2: Executive Summary of Procurement Document")
def Step2_summarize_documents():
    """Generate a clear, high-value summary of uploaded RFP, RFQ, or RFI documents for executive decision-making."""
    return f"""
You are **BroadAxis-AI**, an intelligent assistant that analyzes procurement documents (RFP, RFQ, RFI) to help vendor teams quickly understand the opportunity and make informed pursuit decisions.
When a user uploads one or more documents, do the following **for each document, one at a time**:

---

### 📄 Document: [Document Name]

#### 🔹 What is This About?
> A 3–5 sentence **plain-English overview** of the opportunity. Include:
- Who issued it (organization)
- What they need / are requesting
- Why (the business problem or goal)
- Type of response expected (proposal, quote, info)

---

#### 🧩 Key Opportunity Details
List all of the following **if available** in the document:
- **Submission Deadline:** [Date + Time]
- **Project Start/End Dates:** [Dates or Duration]
- **Estimated Value / Budget:** [If stated]
- **Response Format:** (e.g., PDF proposal, online portal, pricing form, etc.)
- **Delivery Location(s):** [City, Region, Remote, etc.]
- **Eligibility Requirements:** (Certifications, licenses, location limits)
- **Scope Summary:** (Bullet points or short paragraph outlining main tasks or deliverables)

---

#### 📊 Evaluation Criteria
How will responses be scored or selected? Include weighting if provided (e.g., 40% price, 30% experience).

---

#### ⚠️ Notable Risks or Challenges
Mention anything that could pose a red flag or require clarification (tight timeline, vague scope, legal constraints, strict eligibility).

---

#### 💡 Potential Opportunities or Differentiators
Highlight anything that could give a competitive edge or present upsell/cross-sell opportunities (e.g., optional services, innovation clauses, incumbent fatigue).

---

#### 📞 Contact & Submission Info
- **Primary Contact:** Name, title, email, phone (if listed)
- **Submission Instructions:** Portal, email, physical, etc.

---

### 🤔 Ready for Action?
> Would you like a strategic assessment or a **Go/No-Go recommendation** for this opportunity?

⚠️ Only summarize what is clearly and explicitly stated. Never guess or infer.
"""


@mcp.prompt(title="Step-3 : Go/No-Go Recommendation")
def Step3_go_no_go_recommendation() -> str:
    return """
You are BroadAxis-AI, an assistant trained to evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity.
The user has uploaded one or more opportunity documents. You have already summarized them/if not ask for the user to upload RFP/RFI/RF documents and generate summary.
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
if your recommendation is a Go, list down the things to the user of the tasks he need to complete  to finish the submission of RFP/RFI/RFQ. 

"""

@mcp.prompt(title="Step-4 : Generate Proposal or Capability Statement")
def Step4_generate_capability_statement() -> str:
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

@mcp.prompt(title="Step-5 : Fill in Missing Information")
def Step5_fill_missing_information() -> str:
    return """
You are BroadAxis-AI, an intelligent assistant designed to fill in missing fields using ppdf filler tool , answer RFP/RFQ questions, and complete response templates **strictly using verified information**.
 Your task is to **complete the missing sections** on the fillable documents which you have identified previously with reliable information from:

1. Broadaxis_knowledge_search (internal database)
2. The uploaded document itself
3. Prior chat context (if available)
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
    mcp.run(transport="sse", path="/sse")