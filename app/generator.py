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

def generate_fs_from_requirement(
    requirement: str,
    fs_template: str
) -> str:
    """
    Given requirement, and a template (provided by the user each call),
    use RAG retrieval of template and generate a doc strictly following it.
    """
    retriever = build_template_retriever(fs_template)
    retrieved_docs = retriever.get_relevant_documents(requirement)
    # Should always retrieve the template, since it's the only doc.
    retrieved_template = "\n\n".join(doc.page_content for doc in retrieved_docs)
    if not retrieved_template.strip():
        return "No functional specification template found in RAG."

    prompt_template = ChatPromptTemplate.from_template(
       "You are an SAP Functional Consultant. Strictly use the TEMPLATE structure and headings provided below.\n\n"
    "REQUIREMENT:\n{requirement}\n\n"
    "TEMPLATE (STRICTLY FOLLOW):\n{fs_template}\n\n"
    "Write a Functional Specification Document for business stakeholders, strictly following the TEMPLATE headings and order.\n\n"
    "⚠️ Format all headings with hierarchical numbering (e.g., 1., 1.1., 2.1.1, etc.).\n"
    "⚠️ Do NOT use Markdown headings (no #, ##, etc.).\n"
    "⚠️ Do NOT miss any headings and sub heading (if empty mark it NULL or NA).\n"
    "⚠️ Use tables where appropriate: format as grid tables using tabs or clear alignment, compatible with MS Word.\n\n"
    "Ensure the document is professional, readable, and copy-paste ready for MS Word formatting."
    )
    messages = prompt_template.format_messages(
        requirement=requirement,
        fs_template=retrieved_template
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


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
        "You are an SAP Technical Consultant. Strictly use the TEMPLATE structure and headings provided below.\n\n"
    "REQUIREMENT:\n{requirement}\n\n"
    "TEMPLATE (STRICTLY FOLLOW):\n{ts_template}\n\n"
    "Write a Technical Specification Document for business stakeholders, strictly following the TEMPLATE headings and order.\n\n"
    "⚠️ Format all headings with hierarchical numbering (e.g., 1., 1.1., 2.1.1, etc.).\n"
    "⚠️ Do NOT use Markdown headings (no #, ##, etc.).\n"
    "⚠️ Do NOT miss any headings and sub heading (if empty mark it NULL or NA).\n"
    "⚠️ Use tables where appropriate: format as grid tables using tabs or clear alignment, compatible with MS Word.\n\n"
    "Ensure the document is professional, readable, and copy-paste ready for MS Word formatting."
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
        "You are an expert ABAP developer. Strictly use the TEMPLATE structure and comments provided below as a template.\n\n"
    "REQUIREMENT:\n{requirement}\n\n"
    "ABAP CODE TEMPLATE (STRICTLY FOLLOW):\n{abap_template}\n\n"
    "Write production-ready, well-commented ABAP code to fulfill the REQUIREMENT above. "
    "Strictly follow the CODE TEMPLATE structure, style, headers, and comments. "
    "Do not add or omit code sections that are in the template. Use modern, readable ABAP and include meaningful comments. "
    "Only output the ABAP code. Do not use Markdown or code fences."
    )
    messages = prompt_template.format_messages(
        requirement=requirement,
        abap_template=retrieved_template
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)