import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()


def build_template_retriever(template: str):
    """Build a retriever for the passed-in template string."""
    documents = [Document(page_content=template)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embedding)
    return vectorstore.as_retriever()


def generate_ts_from_requirement(
    requirement: str,
    ts_template: str
) -> str:
    """
    Given user requirement and a template (provided per call),
    use RAG retrieval of template and generate a technical specification strictly following it.
    """
    retriever = build_template_retriever(ts_template)
    retrieved_docs = retriever.get_relevant_documents(requirement)
    # Should always retrieve the template, since it's the only doc.
    retrieved_template = "\n\n".join(doc.page_content for doc in retrieved_docs)
    if not retrieved_template.strip():
        return "No technical specification template found in RAG."

    prompt_template = ChatPromptTemplate.from_template(
    "You are an experienced SAP Technical Architect. Your task is to create a detailed Technical Design Document (TDD) for developers, "
    "based on the REQUIREMENT below. Strictly follow the structure and headings of the provided TEMPLATE. Every section must be included, "
    "using the exact section numbering, titles, and subheadings as defined in the TEMPLATE.\n\n"
    
    "REQUIREMENT:\n{requirement}\n\n"
    
    "TECHNICAL DESIGN DOCUMENT TEMPLATE (STRICTLY FOLLOW):\n{ts_template}\n\n"
    
    "INSTRUCTIONS:\n"
    "1. Use precise technical terminology relevant to SAP ABAP, SAP RAP, CDS views, behavior definitions, projection views, tables, structures, fields, actions, enhancements, and integrations as appropriate to the business requirement.\n"
    "2. Each section must be actionable, detailed, and directly usable by a developer to implement the solution.\n"
    "3. Provide clear logic description, including process flow, data model relationships, entity keys, joins, associations, and method responsibilities where relevant.\n"
    "4. For validations, actions, and business logic, explain conditions, invocation points, and error handling mechanisms in the TDD section.\n"
    "5. Specify RAP constructs in context (e.g., managed/unmanaged scenarios, drafts, EML usage, RAP handler methods).\n"
    "6. When interface or integration logic is involved, include interface details, data mapping, and external system touchpoints.\n"
    "7. Use structured, hierarchical numbering for all headings (e.g., 1., 1.1., 2.1.1, etc.).\n"
    "8. Do not use Markdown formatting (no #, ##, etc.)—use plain text, compatible with MS Word and SAP documentation standards.\n"
    "9. Do not omit or rename any heading or subheading from the template—even if content is not applicable, populate with 'NA'.\n"
    "10. Use tabular formatting for entity fields, keys, and mappings, using tabs or clear cell alignment for easy copy-paste to MS Word.\n\n"
    
    "Deliver a professional, fully detailed, and implementation-ready Technical Design Document. Output ONLY the completed TDD in plain text—do not provide any explanations, formatting notes, or Markdown."
)
    messages = prompt_template.format_messages(
        requirement=requirement,
        ts_template=retrieved_template
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def generate_abap_code_from_requirement(
    requirement: str,
    abap_template: str
) -> str:
    """
    Given a requirement and an ABAP template,
    use RAG retrieval of template and generate ABAP code strictly following it.
    """
    retriever = build_template_retriever(abap_template)
    retrieved_docs = retriever.get_relevant_documents(requirement)
    # Should always retrieve the template, since it's the only doc.
    retrieved_template = "\n\n".join(doc.page_content for doc in retrieved_docs)
    if not retrieved_template.strip():
        return "No ABAP template found in RAG."

    prompt_template = ChatPromptTemplate.from_template(
    "You are an expert ABAP RAP (Restful ABAP Programming Model) developer. "
    "You are responsible for delivering a complete, production-grade RAP implementation."
    " Strictly use the TEMPLATE structure, sections, RAP framework annotations, and comments provided below as your authoritative blueprint.\n\n"

    "REQUIREMENT:\n{requirement}\n\n"

    "RAP ABAP CODE TEMPLATE (STRICTLY FOLLOW):\n{abap_template}\n\n"

    "INSTRUCTIONS:"
    "\n- Your output must include all necessary RAP artifacts to fulfill the requirement: "
    "the RAP handler class, the root CDS view, child CDS view(s), projection CDS view(s), behavior definition, draft table definition, associations, and any other objects required for a functional ABAP RAP solution (as prescribed by the template)."
    "\n- Strictly comply with the TEMPLATE's structure, code layout, RAP managed/unmanaged implementation patterns, annotations, CDS view/entity design, behavior definitions, class implementation sections, method headers, and all template comments."
    "\n- Do not add, omit, rename, or re-order code sections not present in the template. Use only sanctioned modern, idiomatic ABAP and RAP best practices."
    "\n- Ensure the code is end-to-end RAP-ready and uses only sections present in the template, including all provided code segments such as: "
    "CDS view syntax, behavior definition, handler class (with all method skeletons), and associations."
    "\n- If the template includes comment placeholders, fill them with appropriate ABAP code per the requirement."
    "\n- Do NOT use Markdown or code fence formatting. Do NOT add any explanations or extra text."
    "\n- The response MUST be ABAP code only, precisely and exclusively following the structure of the RAP CODE TEMPLATE."
)
    messages = prompt_template.format_messages(
        requirement=requirement,
        abap_template=retrieved_template
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)