# -*- coding: utf-8 -*-
"""
train_eachclass.py
- Faster R-CNN (ResNet50 FPN v2, COCO 사전학습) / 멀티 COCO 데이터셋
- 학습은 합치고, 평가는 데이터셋별 분리
- ✅ Resume(안전 허용 목록 기반), ✅ AMP 최신 API(torch.amp), ✅ 재현성, ✅ TensorBoard, ✅ Weights & Biases
- ✅ Per-class metrics (accuracy_img, precision/recall, mean IoU of TPs, R²(cx,cy,w,h)) to W&B Table
- ✅ Normalize 이중 적용 제거
- ✅ 작은 회전 + Flip augmentation
- ✅ 멀티 스케일(min_size 리스트)
- ✅ AdamW + epoch 기반 학습
- ✅ class-balanced sampler (희귀 클래스 oversampling)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 디버깅용, 성능 저하

import time
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy  # <- resume 허용목록에 필요
import torch
import torch.serialization  # <- resume 허용목록에 필요
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.utils.data import WeightedRandomSampler

import torchvision
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ✅ 최신 권장 AMP API (경고 제거)
from torch.amp import GradScaler, autocast

# === W&B ===
import wandb
import random

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# ------------------------------
# 경로 설정
# ------------------------------
TRAIN_RAPISCAN_ROOT = Path(r"D:/xray/Rapiscan")
TRAIN_RAPISCAN_JSON = Path(r"D:/xray/Annotation/Train/CoCo/coco_rapiscan_fixed.json")
TRAIN_SMITH_ROOT    = Path(r"D:/xray/Smith")
TRAIN_SMITH_JSON    = Path(r"D:/xray/Annotation/Train/CoCo/coco_smith_fixed.json")
TRAIN_ASTRO_ROOT    = Path(r"D:/xray/Astrophysics")
TRAIN_ASTRO_JSON    = Path(r"D:/xray/Annotation/Train/CoCo/coco_astrophysics_fixed.json")

TEST_RAPISCAN_ROOT  = Path(r"D:/xray/Eval/Rapiscan")
TEST_RAPISCAN_JSON  = Path(r"D:/xray/Annotation/Label/CoCo/coco_eval_rapiscan_fixed.json")
TEST_SMITH_ROOT     = Path(r"D:/xray/Eval/Smith")
TEST_SMITH_JSON     = Path(r"D:/xray/Annotation/Label/CoCo/coco_eval_smith_fixed.json")
TEST_ASTRO_ROOT     = Path(r"D:/xray/Eval/Astrophysics")
TEST_ASTRO_JSON     = Path(r"D:/xray/Annotation/Label/CoCo/coco_eval_astrophysics_fixed.json")

# ------------------------------
# 전역 카테고리 맵(name→global_id)
# ------------------------------
ALL_JSON_FILES = [
    TRAIN_RAPISCAN_JSON, TRAIN_SMITH_JSON, TRAIN_ASTRO_JSON,
    TEST_RAPISCAN_JSON, TEST_SMITH_JSON, TEST_ASTRO_JSON
]
GLOBAL_NAME2ID: Dict[str, int] = {}
_gid = 1
print("--- Creating Global Category Map ---")
for jf in ALL_JSON_FILES:
    if jf.exists():
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "categories" in data:
                for c in sorted(data["categories"], key=lambda x: x.get("name","")):
                    n = c.get("name")
                    if n and n not in GLOBAL_NAME2ID:
                        GLOBAL_NAME2ID[n] = _gid
                        _gid += 1
            print(f"Processed categories from: {jf.name}")
        except Exception as e:
            print(f"ERROR reading {jf}: {e}")
    else:
        print(f"Warning: missing {jf}")
NUM_OBJECT = len(GLOBAL_NAME2ID)
if NUM_OBJECT == 0:
    raise ValueError("No categories found.")
NUM_CLASSES = NUM_OBJECT + 1  # 0=background
GLOBAL_ID2NAME = {v:k for k,v in GLOBAL_NAME2ID.items()}
print(f"Global map created. Unique object categories: {NUM_OBJECT}")
print(f"NUM_CLASSES set to: {NUM_CLASSES}")

# ------------------------------
# 하이퍼파라미터/디바이스 & 실행 옵션
# ------------------------------
NUM_EPOCHS      = 40       # ✅ 에폭 기반 학습, 이걸로 학습 더 돌리기
NUM_ITERATIONS  = 0         # ✅ 0이면 iteration 기반 상한 없음
BATCH_SIZE      = 16

# AdamW 하이퍼파라미터 (필요 시 W&B sweep 권장)
BASE_LR         = 2e-4
WEIGHT_DECAY    = 1e-4
SPLIT           = [0.8, 0.2]  # train:val
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR_BASE    = Path("C:/Users/aicoss/.spyder-py3/runs/hazard_detection")
RUN_NAME        = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR         = LOG_DIR_BASE / RUN_NAME
CHECKPOINT_FILE = Path(r"D:/xray/testCode/checkpoint_model/checkpoint.pth")
CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR  = Path(r"D:/xray/testCode/trained_model")
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESUME          = True         # ✅ 이어학습
USE_AMP         = True         # ✅ 자동 혼합정밀
SEED            = 42           # ✅ 재현성

# per-class custom metrics에서 사용할 score threshold (precision 안정)
PERCLASS_SCORE_THR = 0.5

# === W&B ===
WANDB_PROJECT = "hazard-detection"  # 프로젝트 이름
WANDB_ENTITY  = None                # 팀/워크스페이스 사용시 "your-team"
WANDB_MODE    = "online"            # 내부망/오프라인이면 "offline"

print(f"사용 장치: {DEVICE}")
print(f"TensorBoard 로그 디렉토리: {LOG_DIR}")

# ------------------------------
# 유틸
# ------------------------------
def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

class RandomRotateSmall:
    """
    X-ray용 소규모 회전 augmentation (±degrees).
    bbox를 회전에 맞춰 재계산하지 않기 때문에, degrees는 작게 유지.
    """
    def __init__(self, degrees=5.0):
        self.degrees = float(degrees)

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img, angle)

def get_transform(train: bool):
    """
    ✅ Normalize 제거 (Faster R-CNN 내부에서 이미 mean/std normalize 수행)
    ✅ train 시에만 수평 플립 + 소규모 회전 추가
    """
    ts = [T.ToTensor()]
    if train:
        ts.append(T.RandomHorizontalFlip(0.5))
        ts.append(RandomRotateSmall(degrees=5.0))
    return T.Compose(ts)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    try:
        return tuple(zip(*batch))
    except Exception:
        return None, None

# ------------------------------
# Per-class custom metrics helpers
# ------------------------------
def _box_iou_np(a, b):
    # a: [Na,4], b: [Nb,4] in xyxy
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], a.shape[0] if b.size == 0 else b.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = a[:,0], a[:,1], a[:,2], a[:,3]
    x21, y21, x22, y2b = b[:,0], b[:,1], b[:,2], b[:,3]
    xa = np.maximum(x11[:,None], x21[None,:])
    ya = np.maximum(y11[:,None], y21[None,:])
    xb = np.minimum(x12[:,None], x22[None,:])
    yb = np.minimum(y12[:,None], y2b[None,:])
    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area_a = np.clip(x12 - x11, 0, None) * np.clip(y12 - y11, 0, None)
    area_b = np.clip(x22 - x21, 0, None) * np.clip(y2b - y21, 0, None)
    union = area_a[:,None] + area_b[None,:] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)

def _xyxy_to_cxcywh(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0,4), dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    w = np.clip(x2 - x1, 0, None)
    h = np.clip(y2 - y1, 0, None)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

def _match_per_class(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5):
    """
    간단 그리디 매칭: pred score 내림차순으로 IoU>=thr인 GT 하나와 1:1 매칭
    return: list of (g_idx, p_idx, iou)
    """
    gt_boxes  = np.asarray(gt_boxes, dtype=np.float32)
    pred_boxes  = np.asarray(pred_boxes, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return []
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    ious = _box_iou_np(gt_boxes, pred_boxes)  # [Ng, Np]
    matched = []
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    pred_used = np.zeros(len(pred_boxes), dtype=bool)
    for p in range(len(pred_boxes)):
        if ious.shape[0] == 0:
            break
        gi = int(np.argmax(ious[:, p]))
        iou = float(ious[gi, p])
        if (not gt_used[gi]) and (not pred_used[p]) and (iou >= iou_thr):
            gt_used[gi] = True
            pred_used[p] = True
            matched.append((gi, p, iou))
    out = []
    for gi, p_sorted, iou in matched:
        p_idx = int(order[p_sorted])
        out.append((gi, p_idx, iou))
    return out

def _r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:,None]; y_pred = y_pred[:,None]
    if y_true.shape[0] == 0:
        return [np.nan]*y_true.shape[1]
    r2_list = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:,j], y_pred[:,j]
        ss_res = float(np.sum((yt - yp)**2))
        ss_tot = float(np.sum((yt - np.mean(yt))**2))
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        r2_list.append(r2)
    return r2_list

class PerClassAccumulator:
    """
    - accuracy_img: 이미지 단위 존재/부재 정확도
    - precision/recall: IoU>=thr 매칭 TP/FP/FN 기반
    - mean_IoU_TP: 매칭된 TP IoU 평균
    - R²: TP들에 대해 (cx, cy, w, h) 회귀 적합도
    """
    def __init__(self, name2gid: dict, iou_thr=0.5, score_thr=0.05):
        self.name2gid = name2gid
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.K = max(name2gid.values()) + 1  # background 0 포함
        self.tp = np.zeros(self.K, dtype=np.int64)
        self.fp = np.zeros(self.K, dtype=np.int64)
        self.fn = np.zeros(self.K, dtype=np.int64)
        self.iou_sum = np.zeros(self.K, dtype=np.float64)
        self.iou_cnt = np.zeros(self.K, dtype=np.int64)
        self.bbox_true = {k: [] for k in range(self.K)}
        self.bbox_pred = {k: [] for k in range(self.K)}
        self.img_tp = np.zeros(self.K, dtype=np.int64)
        self.img_fp = np.zeros(self.K, dtype=np.int64)
        self.img_tn = np.zeros(self.K, dtype=np.int64)
        self.img_fn = np.zeros(self.K, dtype=np.int64)

    def add_image(self, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores):
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        pred_scores = np.asarray(pred_scores, dtype=np.float32)

        # 이미지 레벨 존재/부재 정확도 집계
        for k in range(1, self.K):
            gt_present = np.any(gt_labels == k)
            pred_present = np.any((pred_labels == k) & (pred_scores >= self.score_thr))
            if   gt_present and pred_present: self.img_tp[k] += 1
            elif gt_present and not pred_present: self.img_fn[k] += 1
            elif (not gt_present) and pred_present: self.img_fp[k] += 1
            else: self.img_tn[k] += 1

        # 클래스별 매칭/집계
        for k in range(1, self.K):
            g = gt_boxes[gt_labels == k]
            mask = (pred_labels == k) & (pred_scores >= self.score_thr)
            p = pred_boxes[mask]
            s = pred_scores[mask]

            matched = _match_per_class(g, p, s, iou_thr=self.iou_thr)
            matched_g = [mg for mg,_,_ in matched]
            matched_p = [mp for _,mp,_ in matched]

            tp_k = len(matched)
            fp_k = len(p) - len(set(matched_p))
            fn_k = len(g) - len(set(matched_g))

            self.tp[k] += tp_k
            self.fp[k] += fp_k
            self.fn[k] += fn_k

            for gi, pi, iou in matched:
                self.iou_sum[k] += float(iou)
                self.iou_cnt[k] += 1

            if tp_k > 0:
                g_cxcywh = _xyxy_to_cxcywh(g[matched_g]) if len(matched_g) > 0 else np.zeros((0,4), dtype=np.float32)
                p_cxcywh = _xyxy_to_cxcywh(p[matched_p]) if len(matched_p) > 0 else np.zeros((0,4), dtype=np.float32)
                for row_t, row_p in zip(g_cxcywh, p_cxcywh):
                    self.bbox_true[k].append(row_t)
                    self.bbox_pred[k].append(row_p)

    def summarize_rows(self, gid2name: dict):
        rows = []
        for k in range(1, self.K):
            prec = float(self.tp[k]) / float(self.tp[k] + self.fp[k]) if (self.tp[k]+self.fp[k])>0 else np.nan
            rec  = float(self.tp[k]) / float(self.tp[k] + self.fn[k]) if (self.tp[k]+self.fn[k])>0 else np.nan
            acc_img = float(self.img_tp[k] + self.img_tn[k]) / float(self.img_tp[k] + self.img_fp[k] + self.img_tn[k] + self.img_fn[k]) if (self.img_tp[k] + self.img_fp[k] + self.img_tn[k] + self.img_fn[k])>0 else np.nan
            miou = (self.iou_sum[k] / self.iou_cnt[k]) if self.iou_cnt[k] > 0 else np.nan

            if len(self.bbox_true[k]) >= 2:
                y_true = np.asarray(self.bbox_true[k], dtype=np.float32)
                y_pred = np.asarray(self.bbox_pred[k], dtype=np.float32)
                r2_cx, r2_cy, r2_w, r2_h = _r2_score(y_true, y_pred)
            else:
                r2_cx = r2_cy = r2_w = r2_h = np.nan

            rows.append({
                "category": gid2name.get(k, f"id{k}"),
                "gid": k,
                "precision": prec,
                "recall": rec,
                "accuracy_img": acc_img,
                "mean_IoU_TP": miou,
                "R2_cx": r2_cx, "R2_cy": r2_cy, "R2_w": r2_w, "R2_h": r2_h,
                "TP": int(self.tp[k]), "FP": int(self.fp[k]), "FN": int(self.fn[k]),
                "TN": int(self.img_tn[k]),
                "img_TP": int(self.img_tp[k]), "img_FP": int(self.img_fp[k]),
                "img_TN": int(self.img_tn[k]), "img_FN": int(self.img_fn[k]),
            })
        return rows

def _log_det_metrics_table(rows, ds_name, step, sort_by="precision", writer=None):
    # W&B Table
    try:
        table = wandb.Table(columns=[
            "category","precision","recall","accuracy_img","mean_IoU_TP",
            "R2_cx","R2_cy","R2_w","R2_h",
            "TP","FP","FN","TN",
            "img_TP","img_FP","img_TN","img_FN"
        ])
        def to_num(v): 
            return None if (v is None or (isinstance(v,float) and np.isnan(v))) else float(v)
        key = (lambda r: (-1 if (r.get(sort_by) is None or (isinstance(r.get(sort_by), float) and np.isnan(r.get(sort_by)))) else r.get(sort_by)))
        for r in sorted(rows, key=key, reverse=True):
            table.add_data(
                r["category"],
                to_num(r["precision"]),
                to_num(r["recall"]),
                to_num(r["accuracy_img"]),
                to_num(r["mean_IoU_TP"]),
                to_num(r["R2_cx"]),
                to_num(r["R2_cy"]),
                to_num(r["R2_w"]),
                to_num(r["R2_h"]),
                r["TP"], r["FP"], r["FN"],
                r["TN"],
                r["img_TP"], r["img_FP"], r["img_TN"], r["img_FN"]
            )
        wandb.log({f"per_class_det/{ds_name}": table}, step=step)
    except Exception as e:
        print(f"[W&B] det metrics table log failed: {e}")

    # (선택) TensorBoard worst10 텍스트
    if writer is not None and len(rows) > 0:
        worst10 = sorted(rows, key=lambda r: (np.nan_to_num(r["precision"], nan=-1.0)))[:10]
        lines = ["| class | prec | rec | acc_img | IoU |",
                 "|---|---:|---:|---:|---:|"]
        for r in worst10:
            fmt = lambda v: "nan" if (v is None or (isinstance(v,float) and np.isnan(v))) else f"{v:.3f}"
            lines.append(f"| {r['category']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['accuracy_img'])} | {fmt(r['mean_IoU_TP'])} |")
        writer.add_text(f"per_class_det/{ds_name}/worst10", "\n".join(lines), step=step)

def _coco_per_class_breakdown(ev, coco_gt, used_img_ids=None):
    precisions = ev.eval['precision']  # [T, R, K, A, M]
    iouThrs   = ev.params.iouThrs
    catIds    = ev.params.catIds or coco_gt.getCatIds()
    cats      = coco_gt.loadCats(catIds)
    area_idx  = 0
    maxdets   = list(ev.params.maxDets)
    m_idx     = maxdets.index(100) if 100 in maxdets else (len(maxdets)-1)
    i50_idx   = int(np.argmin(np.abs(iouThrs - 0.5)))
    rows = []
    for k, c in enumerate(cats):
        name, cid = c["name"], c["id"]
        p_all = precisions[:, :, k, area_idx, m_idx]
        p_all = p_all[p_all > -1]
        ap_5095 = float(np.mean(p_all)) if p_all.size else float('nan')
        p_50 = precisions[i50_idx, :, k, area_idx, m_idx]
        p_50 = p_50[p_50 > -1]
        ap_50 = float(np.mean(p_50)) if p_50.size else float('nan')
        if used_img_ids:
            ann_ids = coco_gt.getAnnIds(imgIds=list(used_img_ids), catIds=[cid])
        else:
            ann_ids = coco_gt.getAnnIds(catIds=[cid])
        n_gt = len(ann_ids)
        rows.append({"category": name, "category_id": cid, "AP50": ap_50, "AP50:95": ap_5095, "nGT": n_gt})
    return rows

def _log_per_class_ap_to_wandb(rows, ds_name, step):
    try:
        table = wandb.Table(columns=["category","AP50","AP50:95","nGT"])
        for r in sorted(rows, key=lambda r: (-1 if np.isnan(r["AP50"]) else r["AP50"]), reverse=True):
            table.add_data(
                r["category"],
                None if np.isnan(r["AP50"]) else float(r["AP50"]),
                None if np.isnan(r["AP50:95"]) else float(r["AP50:95"]),
                int(r["nGT"]),
            )
        wandb.log({f"per_class_ap/{ds_name}": table}, step=step)
    except Exception as e:
        print(f"[W&B] per-class AP table log failed: {e}")

# ------------------------------
# COCO Dataset
# ------------------------------
class XrayHazardDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, global_cat_map=None):
        self.root = Path(root)
        self.transforms = transforms
        self.global_name2id = global_cat_map
        if self.global_name2id is None:
            raise ValueError("global_cat_map missing")

        print(f"Loading annotation: {annotation_file}")
        self.coco = COCO(str(annotation_file))
        print("Loaded.")

        all_ids = sorted(self.coco.getImgIds())
        self.img_ids = [i for i in all_ids if self.coco.getAnnIds(imgIds=i)]
        local_cats = self.coco.loadCats(self.coco.getCatIds())
        self.local_id2name = {c["id"]: c.get("name", f"Unnamed_{c['id']}") for c in local_cats}
        print(f"  File has {len(local_cats)} local categories.")
        print(f"Dataset init done. Usable images: {len(self.img_ids)}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        try:
            info = self.coco.loadImgs(img_id)[0]
            path = self.root / info["file_name"]
            if not path.is_file():
                return None
            from PIL import Image
            image = Image.open(str(path)).convert("RGB")

            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            boxes, labels = [], []
            for a in anns:
                if "bbox" not in a or len(a["bbox"]) != 4:
                    continue
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                x2, y2 = x + w, y + h
                name = self.local_id2name.get(a.get("category_id"))
                if not name:
                    continue
                gid = self.global_name2id.get(name)
                if gid is None or gid < 1 or gid >= NUM_CLASSES:
                    continue
                boxes.append([x, y, x2, y2])
                labels.append(gid)

            if not boxes:
                return None
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([img_id]),
            }
            img_t = self.transforms(image) if self.transforms else T.ToTensor()(image)
            return img_t, target
        except Exception:
            return None

# ------------------------------
# 모델(사전학습) 생성
# ------------------------------
def create_model(num_classes: int):
    print("--- create_model (pretrained Faster R-CNN R50-FPN v2) ---")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT  # COCO pretrain
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Detection head 교체
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    print(f"  Head out_features -> {num_classes}")

    # ✅ 멀티 스케일 트레이닝: min_size 리스트 사용
    model.transform.min_size = [600, 800, 1000]  # 기본은 [800]
    model.transform.max_size = 1333              # 필요 시 1024로 조정 가능

    return model

# ------------------------------
# COCO 평가 유틸
# ------------------------------
def _ensure_coco_info(coco: COCO):
    if "info" not in coco.dataset:
        coco.dataset["info"] = {"description": "eval", "version": "1.0", "year": 2025}

@torch.no_grad()
def evaluate_single_dataset(model, loader, device, writer=None, it=0, ds_name=""):
    model.eval()
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    if not hasattr(ds, "coco"):
        print(f"[{ds_name}] no COCO GT")
        return 0.0
    coco_gt = ds.coco
    _ensure_coco_info(coco_gt)

    gid2name = {v: k for k, v in ds.global_name2id.items()}
    local_id2name_gt = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    name2coco = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    results: List[Dict] = []
    used = set()

    # per-class metrics 누적기
    det_accum = PerClassAccumulator(ds.global_name2id, iou_thr=0.5, score_thr=PERCLASS_SCORE_THR)

    print(f"[{ds_name}] Starting COCO evaluation loop...")
    for i, batch in enumerate(tqdm(loader, desc=f"[{ds_name}] Evaluating")):
        if batch is None or batch[0] is None:
            continue
        images, targets = batch
        if not images:
            continue
        images = [img.to(device) for img in images]
        try:
            outputs = model(images)
        except Exception as e:
            print(f"[{ds_name}] inference error: {e}")
            continue
        outs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]
        tgts_id_only = [{k: v.to("cpu") for k, v in t.items() if k == "image_id"} for t in targets]

        for t_id, o in zip(tgts_id_only, outs):
            if "image_id" not in t_id:
                continue
            img_id = int(t_id["image_id"].item())
            used.add(img_id)

            p_boxes  = o.get("boxes",  torch.empty(0)).numpy().astype(np.float32)
            p_labels = o.get("labels", torch.empty(0, dtype=torch.long)).numpy().astype(np.int64)
            p_scores = o.get("scores", torch.empty(0)).numpy().astype(np.float32)

            for (x1, y1, x2, y2), lab, sc in zip(p_boxes, p_labels, p_scores):
                name = gid2name.get(int(lab))
                cid = name2coco.get(name)
                w = float(x2 - x1); h = float(y2 - y1)
                if cid is None or w <= 0 or h <= 0:
                    continue
                results.append({
                    "image_id": img_id,
                    "category_id": int(cid),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(sc)
                })

            # GT
            g_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
            g_boxes, g_labels = [], []
            for a in g_anns:
                if "bbox" not in a:
                    continue
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                cname = local_id2name_gt.get(a["category_id"])
                gid = ds.global_name2id.get(cname, None)
                if gid is None:
                    continue
                g_boxes.append([x, y, x+w, y+h])
                g_labels.append(gid)
            g_boxes = np.array(g_boxes, dtype=np.float32)
            g_labels = np.array(g_labels, dtype=np.int64)

            # custom metrics 누적
            det_accum.add_image(g_boxes, g_labels, p_boxes, p_labels, p_scores)

    if not results:
        print(f"[{ds_name}] no detections -> mAP 0.0")
        det_rows = det_accum.summarize_rows(gid2name)
        _log_det_metrics_table(det_rows, ds_name, it, writer)
        wandb.log({f"mAP/{ds_name}_0.5": 0.0, f"mAP/{ds_name}_0.5:0.95": 0.0}, step=it)
        return 0.0

    try:
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if used:
            ev.params.imgIds = sorted(list(used))
        ev.evaluate(); ev.accumulate(); ev.summarize()
        m05 = float(ev.stats[1])

        ap_rows = _coco_per_class_breakdown(ev, coco_gt, used_img_ids=used)
        _log_per_class_ap_to_wandb(ap_rows, ds_name, it)

        det_rows = det_accum.summarize_rows(gid2name)
        _log_det_metrics_table(det_rows, ds_name, it, writer)

        if writer:
            writer.add_scalar(f"mAP/{ds_name}_0.5", m05, it)
            writer.add_scalar(f"mAP/{ds_name}_0.5:0.95", float(ev.stats[0]), it)
        wandb.log({
            f"mAP/{ds_name}_0.5": m05,
            f"mAP/{ds_name}_0.5:0.95": float(ev.stats[0]),
        }, step=it)
        return m05
    except Exception as e:
        print(f"[{ds_name}] eval error: {e}")
        det_rows = det_accum.summarize_rows(gid2name)
        _log_det_metrics_table(det_rows, ds_name, it, writer)
        wandb.log({f"mAP/{ds_name}_0.5": 0.0, f"mAP/{ds_name}_0.5:0.95": 0.0}, step=it)
        return 0.0

# ------------------------------
# class-balanced sampler weights
# ------------------------------
def _build_class_balanced_weights(train_parts, global_name2id):
    """
    train_parts: random_split으로 나뉜 Subset 리스트
    - 각 Subset.dataset: XrayHazardDataset
    - 각 Subset.indices: base dataset 내 인덱스 목록

    전략:
      1) 각 이미지에 포함된 global class 집합을 COCO annotation으로 계산 (이미지는 안 엶)
      2) 클래스별 등장 이미지 수를 세고 (class_counts)
      3) 이미지 weight = (해당 이미지의 클래스들에 대해 1/count 의 평균)
    """
    K = max(global_name2id.values()) + 1
    class_counts = np.zeros(K, dtype=np.int64)

    subset_img_classes_list = []

    # 1-pass: class_counts 계산
    for subset in train_parts:
        base_ds = subset.dataset          # XrayHazardDataset
        coco = base_ds.coco
        local_id2name = base_ds.local_id2name

        img_classes_cache = {}
        for local_idx in subset.indices:
            img_id = base_ds.img_ids[local_idx]
            if img_id in img_classes_cache:
                gids = img_classes_cache[img_id]
            else:
                ann_ids = coco.getAnnIds(imgIds=[img_id])
                anns = coco.loadAnns(ann_ids)
                cls_set = set()
                for a in anns:
                    cid = a.get("category_id")
                    name = local_id2name.get(cid)
                    if not name:
                        continue
                    gid = global_name2id.get(name)
                    if gid is None:
                        continue
                    cls_set.add(gid)
                gids = sorted(cls_set)
                img_classes_cache[img_id] = gids

            for gid in gids:
                class_counts[gid] += 1

        subset_img_classes_list.append(img_classes_cache)

    class_counts = np.where(class_counts == 0, 1, class_counts)

    # 2-pass: 각 샘플 weight 계산
    all_weights = []
    for subset, img_classes_cache in zip(train_parts, subset_img_classes_list):
        base_ds = subset.dataset
        weights = []
        for local_idx in subset.indices:
            img_id = base_ds.img_ids[local_idx]
            gids = img_classes_cache.get(img_id, [])
            if not gids:
                weights.append(1.0)
            else:
                invs = [1.0 / float(class_counts[gid]) for gid in gids]
                weights.append(float(np.mean(invs)))
        all_weights.extend(weights)

    weights_tensor = torch.as_tensor(all_weights, dtype=torch.double)
    print(f"[Sampler] Built class-balanced weights, len={len(weights_tensor)}")
    return weights_tensor

# ------------------------------
# 데이터로더 구성
# ------------------------------
def make_loaders():
    print("Initializing Training Datasets...")
    train_src = [
        ("rapiscan_tr", TRAIN_RAPISCAN_ROOT, TRAIN_RAPISCAN_JSON),
        ("smith_tr",    TRAIN_SMITH_ROOT,    TRAIN_SMITH_JSON),
        ("astro_tr",    TRAIN_ASTRO_ROOT,    TRAIN_ASTRO_JSON),
    ]
    test_src = [
        ("rapiscan_te", TEST_RAPISCAN_ROOT,  TEST_RAPISCAN_JSON),
        ("smith_te",    TEST_SMITH_ROOT,     TEST_SMITH_JSON),
        ("astro_te",    TEST_ASTRO_ROOT,     TEST_ASTRO_JSON),
    ]

    train_dsets = []
    for name, root, ann in train_src:
        if not Path(ann).exists():
            continue
        ds = XrayHazardDataset(root, ann, get_transform(True), GLOBAL_NAME2ID)
        if len(ds) > 0:
            ds.name = name
            train_dsets.append(ds)
    if not train_dsets:
        raise ValueError("All training datasets empty")

    train_parts, val_parts = [], []
    for ds in train_dsets:
        n = len(ds)
        val_sz = max(1, int(n * (1.0 - SPLIT[0])))
        tr_sz = n - val_sz
        tr_sub, val_sub = random_split(ds, [tr_sz, val_sz], generator=torch.Generator().manual_seed(SEED))
        val_sub.dataset.name = ds.name
        train_parts.append(tr_sub)
        val_parts.append(val_sub)

    # Concat된 train 세트
    train_concat = ConcatDataset(train_parts)
    print(f"Training Datasets Initialized. Size: {len(train_concat)}")

    # ✅ 클래스 불균형을 고려한 class-balanced sampler
    train_weights = _build_class_balanced_weights(train_parts, GLOBAL_NAME2ID)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    # Val loaders
    val_loaders = {}
    for vs in val_parts:
        nm = getattr(vs.dataset, "name", "val_ds")
        val_loaders[nm] = DataLoader(
            vs, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
            pin_memory=(DEVICE.type == "cuda")
        )

    print("Initializing Test Datasets...")
    test_loaders = {}
    for name, root, ann in test_src:
        if not Path(ann).exists():
            continue
        ds = XrayHazardDataset(root, ann, get_transform(False), GLOBAL_NAME2ID)
        if len(ds) > 0:
            ds.name = name
            test_loaders[name] = DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, collate_fn=collate_fn,
                pin_memory=(DEVICE.type == "cuda")
            )
    print(f"Val loaders: {list(val_loaders.keys())}")
    print(f"Test loaders: {list(test_loaders.keys())}")

    # ✅ 학습은 sampler 사용 (shuffle=False)
    train_loader = DataLoader(
        train_concat,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True
    )
    return train_loader, val_loaders, test_loaders

# ------------------------------
# 학습 루프 (+ Resume + AMP 최신 API + W&B)
# ------------------------------
def start_training():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # === W&B ===
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        mode=WANDB_MODE,
        dir=str(LOG_DIR),
        config=dict(
            NUM_EPOCHS=NUM_EPOCHS,
            NUM_ITERATIONS=NUM_ITERATIONS,
            BATCH_SIZE=BATCH_SIZE,
            BASE_LR=BASE_LR,
            WEIGHT_DECAY=WEIGHT_DECAY,
            SPLIT=SPLIT,
            NUM_CLASSES=NUM_CLASSES,
            USE_AMP=USE_AMP,
            SEED=SEED,
            PERCLASS_SCORE_THR=PERCLASS_SCORE_THR,
        ),
        # 같은 run으로 이어쓰고 싶으면 아래 두 줄 사용:
        # id=RUN_NAME,
        # resume="allow",
    )

    train_loader, val_loaders, test_loaders = make_loaders()

    print("Creating pretrained model...")
    model = create_model(NUM_CLASSES)
    model.to(DEVICE)
    print("Model ready.")

    # === W&B === (모델 파라미터/그래디언트 추적)
    wandb.watch(model, log="all", log_freq=1000)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        print("ERROR: empty train loader")
        wandb.finish()
        return

    total_epochs = NUM_EPOCHS
    warmup_epochs = max(1, int(0.1 * total_epochs))

    def lr_lambda(cur_ep):
        if cur_ep < warmup_epochs:
            return float(cur_ep + 1) / float(max(1, warmup_epochs))
        t = (cur_ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ✅ 최신 AMP API
    scaler = GradScaler('cuda') if (USE_AMP and DEVICE.type == "cuda") else GradScaler('cpu')

    # Resume 상태
    cur_iter, best_map, start_epoch = 0, 0.0, 0
    if RESUME and CHECKPOINT_FILE.exists():
        try:
            print(f"Resuming from {CHECKPOINT_FILE} ...")
            torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
            ckpt = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)

            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  Scheduler state load skipped: {e}")
            best_map = float(ckpt.get("best_map", 0.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            cur_iter = int(ckpt.get("iteration", 0))
            print(f" Resumed: epoch={start_epoch}, iter={cur_iter}, best_map={best_map:.4f}")
        except Exception as e:
            print(f"Resume failed: {e}. Starting fresh.")

    # 학습
    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        if NUM_ITERATIONS > 0 and cur_iter >= NUM_ITERATIONS:
            break
        print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} Training", leave=True)
        for batch in pbar:
            if NUM_ITERATIONS > 0 and cur_iter >= NUM_ITERATIONS:
                break
            if batch is None or batch[0] is None:
                continue
            images, targets = batch
            if not images or not targets:
                continue
            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = [
                {k: v.to(DEVICE, non_blocking=True) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            if (USE_AMP and scaler is not None):
                with autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses):
                    continue
                scaler.scale(losses).backward()
                # (선택) gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses):
                    continue
                losses.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            cur_iter += 1
            pbar.set_postfix(
                loss=f"{float(losses.item()):.4f}",
                iter=f"{cur_iter}",
                lr=f"{optimizer.param_groups[0]['lr']:.1E}"
            )

            if cur_iter % 100 == 0 or cur_iter == 1:
                writer.add_scalar("Loss/train", float(losses.item()), cur_iter)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], cur_iter)
                wandb.log({
                    "Loss/train": float(losses.item()),
                    "LR": optimizer.param_groups[0]['lr'],
                    "iteration": cur_iter,
                    "epoch": epoch + 1,
                }, step=cur_iter)

        scheduler.step()

        # 각 데이터셋별 검증 → 평균
        print("\n--- Starting Validation (per dataset) ---")
        vals = []
        for name, vloader in val_loaders.items():
            m = evaluate_single_dataset(model, vloader, DEVICE, writer, cur_iter, ds_name=name)
            vals.append(m)
        mean_map = float(np.mean(vals)) if vals else 0.0
        print(f"--- Validation Finished --- mean mAP@0.5: {mean_map:.4f}")

        wandb.log({"mAP/val_mean_0.5": mean_map, "epoch": epoch + 1}, step=cur_iter)

        if mean_map > best_map:
            best_map = mean_map
            print(f"🎉 New best mean mAP@0.5: {best_map:.4f}. Saving...")
            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            best_path = BEST_MODEL_DIR / f"best_model_epoch{epoch+1}_map{best_map:.4f}.pth"
            torch.save(model.state_dict(), best_path)

            try:
                art = wandb.Artifact("best_model", type="model")
                art.add_file(str(best_path))
                wandb.log_artifact(art)
            except Exception as e:
                print(f"W&B artifact upload failed: {e}")

        # 체크포인트 저장(원자적 교체)
        ckpt = {
            "epoch": epoch,
            "iteration": cur_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
        }
        tmp = CHECKPOINT_FILE.with_suffix(".pth.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, CHECKPOINT_FILE)

        try:
            ckpt_art = wandb.Artifact("checkpoint", type="model")
            ckpt_art.add_file(str(CHECKPOINT_FILE))
            wandb.log_artifact(ckpt_art)
        except Exception as e:
            print(f"W&B artifact upload failed: {e}")

        if NUM_ITERATIONS > 0 and cur_iter >= NUM_ITERATIONS:
            print("\nTarget iterations reached.")
            break

    print("\n--- Training Finished ---")
    print(f"Total iterations: {cur_iter}")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))}")
    writer.close()

    # 최종 테스트(각 데이터셋별)
    if test_loaders:
        print("\n--- Starting Final Evaluation on Test Set (per dataset) ---")
        model.eval().to(DEVICE)
        test_maps = {}
        for name, tloader in test_loaders.items():
            test_maps[name] = evaluate_single_dataset(model, tloader, DEVICE, None, cur_iter, ds_name=name)

        print("\n--- Final Test Set Evaluation ---")
        for k, v in test_maps.items():
            print(f"{k}: mAP@0.5 = {v:.4f}")
        mean_test = float(np.mean(list(test_maps.values()))) if test_maps else 0.0
        print(f"Mean test mAP@0.5 = {mean_test:.4f}")

        for k, v in test_maps.items():
            wandb.log({f"mAP_test/{k}_0.5": v}, step=cur_iter)
        wandb.log({"mAP_test/mean_0.5": mean_test}, step=cur_iter)
    else:
        print("Skipping final test evaluation (no test loaders).")

    wandb.finish()

# ------------------------------
# 실행
# ------------------------------
if __name__ == "__main__":
    if DEVICE.type == "cuda":
        try:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        print(f"CUDA Version: {torch.version.cuda}")
    start_training()
print("Checkpoint will be saved at:", CHECKPOINT_FILE.resolve())
print("Best model will be saved at:", BEST_MODEL_DIR.resolve())
