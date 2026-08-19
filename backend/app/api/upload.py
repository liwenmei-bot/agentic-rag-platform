"""
文档上传接口。
"""
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.services.rag_service import ingest_document

router = APIRouter()

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"暂不支持的文件类型: {suffix}，目前仅支持 {', '.join(ALLOWED_SUFFIXES)}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / file.filename

    content = await file.read()
    save_path.write_bytes(content)

    try:
        result = ingest_document(str(save_path), file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    return {
        "message": "文档上传并处理成功",
        "doc_id": result["doc_id"],
        "filename": result["filename"],
        "chunk_count": result["chunk_count"],
    }
