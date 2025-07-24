from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
import io
import zipfile
from app.generator import (
    generate_ts_from_requirement,
    generate_fs_from_requirement,
    generate_abap_code_from_requirement,
)
from app.docx_writer import (
    create_functional_spec_docx,
    create_technical_spec_docx,
    create_abap_code_docx,
)

app = FastAPI()

@app.post("/generate-bundle/")
async def generate_fs_ts_abapcode(
    requirement: str = Form(...),
    fs_template: str = Form(...),
    ts_template: str = Form(...),
    abap_template: str = Form(...),
):
    # Generate content for specs
    functional_spec_text = generate_fs_from_requirement(requirement, fs_template)
    technical_spec_text = generate_ts_from_requirement(requirement, ts_template)
    abap_code_text = generate_abap_code_from_requirement(requirement, abap_template)

    # Create each DOCX in memory
    fs_doc = io.BytesIO()
    create_functional_spec_docx(functional_spec_text, fs_doc)
    fs_doc.seek(0)

    ts_doc = io.BytesIO()
    create_technical_spec_docx(technical_spec_text, ts_doc)
    ts_doc.seek(0)

    abap_code_doc = io.BytesIO()
    create_abap_code_docx(abap_code_text, abap_code_doc)
    abap_code_doc.seek(0)

    # Assemble into ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        zipf.writestr("functional_spec.docx", fs_doc.read())
        zipf.writestr("technical_spec.docx", ts_doc.read())
        zipf.writestr("abap_code.docx", abap_code_doc.read())
    zip_buffer.seek(0)

    # Return the ZIP file
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=spec_bundle.zip"}
    )