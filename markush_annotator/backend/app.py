import os
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data_manager import DataManager, graph_to_smiles
from .align_service import AlignService
from .obb_manager import OBBManager

app = FastAPI(title="Markush Visual Annotator API")

# 启用跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_manager = DataManager()
align_service = AlignService()
obb_manager = OBBManager()

# 静态文件路径
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class SaveAnnotationRequest(BaseModel):
    index: int
    esmiles: Optional[str] = ""
    smi: Optional[str] = ""
    image_size: Optional[list] = [800, 600]
    atoms: list
    bonds: list
    sgroups: Optional[list] = []
    brackets: Optional[list] = []
    ocr_atoms: Optional[list] = []
    superatom_merges: Optional[list] = []
    review_status: Optional[str] = "approved"
    review_notes: Optional[str] = ""


class AlignRequest(BaseModel):
    index: int
    atoms: list
    bonds: list
    image_size: list


class CalcSmilesRequest(BaseModel):
    atoms: list
    bonds: list
    brackets: Optional[list] = []


@app.get("/api/info")
def get_info():
    """获取人工池概览统计与过滤标签 (大类归纳)"""
    return data_manager.get_manifest_info()


@app.get("/api/samples")
def list_samples(
    reason: Optional[str] = Query("all"),
    status: Optional[str] = Query("all"),
    keyword: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """条件筛选样本清单"""
    return data_manager.list_samples(
        reason=reason,
        status=status,
        keyword=keyword,
        limit=limit,
        offset=offset
    )


@app.get("/api/sample/{index}")
def get_sample(index: int):
    """获取指定 index 样本的完整图结构与元数据"""
    detail = data_manager.get_sample_detail(index)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Sample #{index} not found")
    return detail


@app.get("/api/image/{index}")
def get_image(index: int):
    """获取原图 PNG 字节流"""
    img_bytes = data_manager.get_image_bytes(index)
    if img_bytes is None:
        raise HTTPException(status_code=404, detail=f"Image for #{index} not found")
    return Response(content=img_bytes, media_type="image/png")


@app.post("/api/save")
def save_annotation(req: SaveAnnotationRequest):
    """保存纠错标注"""
    success = data_manager.save_sample_annotation(req.index, req.model_dump())
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save annotation")
    return {"status": "ok", "index": req.index}


@app.post("/api/calc_smiles")
def calculate_smiles(req: CalcSmilesRequest):
    """根据图结构 (原子与化学键) 调用 RDKit 引擎实时生成 SMILES / CXSMILES"""
    smi = graph_to_smiles(req.atoms, req.bonds, req.brackets)
    return {"status": "ok", "smiles": smi}


@app.post("/api/align")
def align_sample(req: AlignRequest):
    """请求 AI 自动吸附对齐"""
    aligned_atoms = align_service.align_atoms(
        atoms=req.atoms,
        bonds=req.bonds,
        image_shape=req.image_size,
    )
    return {"status": "ok", "atoms": aligned_atoms}


class OcrSuggestRequest(BaseModel):
    index: int
    atoms: list
    bonds: Optional[list] = []


@app.post("/api/sample/{index}/ocr_suggest")
def ocr_suggest_sample(index: int, req: OcrSuggestRequest):
    """请求 AI OCR 智能匹配超原子与 R 基团"""
    img = data_manager.get_cv2_image(index)
    if img is None:
        raise HTTPException(status_code=404, detail=f"Image for #{index} not found")
    new_atoms, updated_count = align_service.ocr_suggest(
        atoms=req.atoms,
        image=img
    )
    return {"status": "ok", "atoms": new_atoms, "updated_count": updated_count}


# =========================================================================
# OBB 目标框标注 API (Oriented Bounding Box Annotation)
# =========================================================================

class SaveOBBAnnotationRequest(BaseModel):
    id: str
    boxes: list
    image_size: Optional[list] = None
    status: Optional[str] = "reviewed"
    annotator: Optional[str] = "human_review"


@app.get("/api/obb/info")
def get_obb_info():
    """获取 OBB 数据集概览与统计"""
    return obb_manager.get_dataset_info()


@app.get("/api/obb/samples")
def list_obb_samples(
    source: str = Query("all"),
    feature: str = Query("all"),
    status: str = Query("all"),
    keyword: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """条件筛选 OBB 样本清单"""
    return obb_manager.list_samples(
        source=source,
        feature=feature,
        status=status,
        keyword=keyword,
        limit=limit,
        offset=offset
    )


@app.get("/api/obb/sample/{sample_id}")
def get_obb_sample(sample_id: str, auto_predict: bool = Query(False)):
    """获取单个样本的元数据与现有 OBB 标注"""
    detail = obb_manager.get_sample_detail(sample_id, auto_predict=auto_predict)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"OBB sample {sample_id} not found")
    return detail


@app.get("/api/obb/image/{sample_id}")
def get_obb_image(sample_id: str):
    """获取 OBB 样本图像"""
    img_bytes = obb_manager.get_image_bytes(sample_id)
    if img_bytes is None:
        raise HTTPException(status_code=404, detail=f"Image for {sample_id} not found")
    return Response(content=img_bytes, media_type="image/png")


@app.post("/api/obb/predict/{sample_id}")
def predict_obb_sample(sample_id: str, conf: float = Query(0.18, ge=0.01, le=1.0)):
    """调用 unified_obb_det 深度模型对图像进行 AI 预标注"""
    try:
        boxes = obb_manager.predict_sample(sample_id, conf=conf)
        return {"status": "ok", "boxes": boxes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/obb/save")
def save_obb_annotation(req: SaveOBBAnnotationRequest):
    """保存 OBB 标注并同步写出 YOLO OBB .txt"""
    success = obb_manager.save_annotation(
        sample_id=req.id,
        boxes=req.boxes,
        image_size=req.image_size,
        status=req.status or "reviewed",
        annotator=req.annotator or "human_review",
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save OBB annotation")
    return {"status": "ok", "id": req.id}


# 挂载前端静态文件
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

