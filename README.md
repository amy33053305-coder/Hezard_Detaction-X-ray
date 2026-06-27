# 🔍 X-Ray Hazard Detection — Faster R-CNN (ResNet50-FPN v2)

AIHub 항만 물류 X-ray 이미지 데이터셋을 기반으로, **Rapiscan · Smith · Astrophysics** 3종의 스캐너 데이터를 통합 학습하고 데이터셋별로 분리 평가하는 객체 탐지 파이프라인입니다.

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [주요 기능](#2-주요-기능)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [환경 설정](#4-환경-설정)
5. [데이터셋 구성](#5-데이터셋-구성)
6. [모델 아키텍처](#6-모델-아키텍처)
7. [학습 설정 (하이퍼파라미터)](#7-학습-설정-하이퍼파라미터)
8. [학습 파이프라인](#8-학습-파이프라인)
9. [평가 지표](#9-평가-지표)
10. [실험 추적 (TensorBoard & W&B)](#10-실험-추적-tensorboard--wb)
11. [체크포인트 및 Resume](#11-체크포인트-및-resume)
12. [실행 방법](#12-실행-방법)
13. [주요 클래스 및 함수 레퍼런스](#13-주요-클래스-및-함수-레퍼런스)
14. [트러블슈팅](#14-트러블슈팅)
15. [향후 개선 방향](#15-향후-개선-방향)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **모델** | Faster R-CNN ResNet50-FPN v2 (COCO 사전학습) |
| **백본** | ResNet-50 + Feature Pyramid Network v2 |
| **입력** | X-ray 이미지 (RGB 변환 후 처리) |
| **출력** | 위험물 클래스별 Bounding Box + Confidence Score |
| **데이터 형식** | COCO JSON annotation |
| **학습 전략** | 멀티 데이터셋 합산 학습 → 데이터셋별 분리 평가 |
| **최적화기** | AdamW |
| **정밀도** | AMP (Automatic Mixed Precision, FP16/FP32 혼합) |

---

## 2. 주요 기능

- ✅ **멀티 COCO 데이터셋 통합** — Rapiscan, Smith, Astrophysics 3종 데이터셋의 카테고리를 전역 ID로 통일
- ✅ **Resume 학습** — 체크포인트에서 안전하게 이어 학습 (원자적 파일 교체 방식)
- ✅ **AMP 최신 API** — `torch.amp.GradScaler` / `autocast` 사용 (구버전 경고 없음)
- ✅ **재현성 보장** — `SEED=42`로 `random`, `numpy`, `torch`, `cudnn` 시드 고정
- ✅ **TensorBoard 로깅** — Loss, LR, mAP 실시간 시각화
- ✅ **Weights & Biases 연동** — 학습 지표, 모델 아티팩트, Per-class 메트릭 Table 업로드
- ✅ **Per-class 커스텀 지표** — 이미지 단위 accuracy, precision/recall, mean IoU(TP), R²(cx/cy/w/h)
- ✅ **데이터 증강** — 수평 플립 (p=0.5) + 소규모 회전 (±5°)
- ✅ **멀티 스케일 학습** — `min_size=[600, 800, 1000]`
- ✅ **클래스 불균형 해소** — `WeightedRandomSampler` 기반 희귀 클래스 오버샘플링
- ✅ **Normalize 이중 적용 방지** — `get_transform()`에서 Normalize 제거 (Faster R-CNN 내부 처리)

---

## 3. 디렉토리 구조

```
D:/xray/
├── Rapiscan/                        # 학습 이미지 (Rapiscan 스캐너)
├── Smith/                           # 학습 이미지 (Smith 스캐너)
├── Astrophysics/                    # 학습 이미지 (Astrophysics 스캐너)
├── Eval/
│   ├── Rapiscan/                    # 평가 이미지 (Rapiscan)
│   ├── Smith/                       # 평가 이미지 (Smith)
│   └── Astrophysics/                # 평가 이미지 (Astrophysics)
├── Annotation/
│   ├── Train/CoCo/
│   │   ├── coco_rapiscan_fixed.json
│   │   ├── coco_smith_fixed.json
│   │   └── coco_astrophysics_fixed.json
│   └── Label/CoCo/
│       ├── coco_eval_rapiscan_fixed.json
│       ├── coco_eval_smith_fixed.json
│       └── coco_eval_astrophysics_fixed.json
└── testCode/
    ├── checkpoint_model/
    │   └── checkpoint.pth           # 이어학습용 체크포인트
    └── trained_model/
        └── best_model_epoch*.pth    # 최고 성능 모델

C:/Users/<user>/.spyder-py3/runs/hazard_detection/
└── <YYYYMMDD-HHMMSS>/               # TensorBoard 로그
```

---

## 4. 환경 설정

### 4.1 요구 사양

| 항목 | 권장 사양 |
|------|-----------|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.9 이상 |
| **GPU** | NVIDIA CUDA 지원 GPU (VRAM 8GB 이상 권장) |
| **CUDA** | 11.7 이상 |
| **RAM** | 16GB 이상 |

### 4.2 패키지 설치

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pycocotools tensorboard wandb tqdm Pillow numpy
```

> **Windows 환경 주의**: `pycocotools` 설치 시 Visual C++ Build Tools가 필요합니다.
> 또는 `pip install pycocotools-windows` 사용을 권장합니다.

### 4.3 W&B 초기 설정

```bash
wandb login
```

---

## 5. 데이터셋 구성

### 5.1 전역 카테고리 맵 생성

스크립트 실행 시 자동으로 6개의 COCO JSON 파일을 순회하여 전역 카테고리 ID를 생성합니다.

```
카테고리명 (name) → 전역 ID (global_id, 1부터 시작)
background → 0 (reserved)
```

- 중복 카테고리는 한 번만 등록됩니다.
- `NUM_CLASSES = 전역 카테고리 수 + 1 (background)`

### 5.2 데이터셋 클래스 (`XrayHazardDataset`)

| 동작 | 설명 |
|------|------|
| annotation 로드 | `pycocotools.COCO`를 통해 COCO JSON 파싱 |
| 유효 이미지 필터링 | annotation이 존재하는 이미지만 사용 |
| bbox 변환 | COCO 형식 `[x, y, w, h]` → `[x1, y1, x2, y2]` (xyxy) |
| 전역 ID 매핑 | local category_id → name → global_id |
| 예외 처리 | 이미지 로드 실패 또는 유효 annotation 없을 시 `None` 반환 |

### 5.3 데이터 분할

| 세트 | 비율 | 설명 |
|------|------|------|
| Train | 80% | 클래스 불균형 보정을 위한 WeightedRandomSampler 적용 |
| Validation | 20% | 데이터셋별로 분리 평가 |
| Test | 별도 | Eval 폴더의 독립 테스트셋 |

---

## 6. 모델 아키텍처

### 6.1 기본 구조

```
입력 이미지
    ↓
[ResNet-50 Backbone]
    ↓
[Feature Pyramid Network v2 (FPN v2)]
    ↓
[Region Proposal Network (RPN)]
    ↓
[RoI Align + Box Head]
    ↓
[FastRCNNPredictor] ← NUM_CLASSES 출력으로 교체
    ↓
클래스별 박스 + 스코어
```

### 6.2 주요 설정

| 항목 | 값 |
|------|-----|
| 사전학습 가중치 | COCO pretrain (`FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`) |
| Detection Head 교체 | `FastRCNNPredictor(in_feats, NUM_CLASSES)` |
| 멀티 스케일 min_size | `[600, 800, 1000]` |
| max_size | `1333` |

### 6.3 내부 전처리 (모델 내부 자동 수행)

Faster R-CNN은 `GeneralizedRCNNTransform`을 통해 내부적으로 ImageNet 평균/표준편차로 normalize합니다.
→ **`get_transform()`에서 별도 Normalize 적용 금지** (이중 적용 방지)

---

## 7. 학습 설정 (하이퍼파라미터)

```python
NUM_EPOCHS     = 40          # 전체 학습 에폭 수
BATCH_SIZE     = 16          # 배치 크기
BASE_LR        = 2e-4        # AdamW 기본 학습률
WEIGHT_DECAY   = 1e-4        # L2 정규화 계수
SPLIT          = [0.8, 0.2]  # train:val 분할 비율
SEED           = 42          # 재현성 시드
USE_AMP        = True        # 자동 혼합 정밀도 사용 여부
RESUME         = True        # 이어학습 여부
PERCLASS_SCORE_THR = 0.5     # Per-class 지표 계산 시 score 임계값
```

### 7.1 학습률 스케줄러

| 구간 | 전략 |
|------|------|
| Warmup (전체 에폭의 10%) | 선형 증가 (`0 → BASE_LR`) |
| 이후 | Cosine Annealing 감소 (`BASE_LR → 0`) |

수식:
- Warmup: `lr = BASE_LR × (epoch+1) / warmup_epochs`
- Cosine: `lr = BASE_LR × 0.5 × (1 + cos(π × t))`, `t ∈ [0,1]`

---

## 8. 학습 파이프라인

### 8.1 전체 흐름

```
[데이터 로드 및 분할]
    ↓
[Class-Balanced WeightedRandomSampler 구성]
    ↓
[사전학습 모델 생성 + Detection Head 교체]
    ↓
[Resume 체크포인트 로드 (RESUME=True)]
    ↓
FOR each epoch:
    ├─ [Train Loop] → AMP forward/backward → AdamW step
    ├─ [LR Scheduler step]
    ├─ [Val 평가 — 데이터셋별 분리]
    ├─ [Best Model 저장 (mean mAP 기준)]
    └─ [Checkpoint 원자적 저장]
↓
[Final Test 평가 — 데이터셋별 분리]
```

### 8.2 데이터 증강

| 증강 기법 | 설정 |
|-----------|------|
| `T.ToTensor()` | 항상 적용 |
| `T.RandomHorizontalFlip(0.5)` | 학습 시만 적용 |
| `RandomRotateSmall(degrees=5.0)` | 학습 시만 적용 (±5° 회전) |
| Normalize | ❌ 미적용 (Faster R-CNN 내부 처리) |

> **주의**: `RandomRotateSmall`은 bbox 좌표를 회전 변환하지 않습니다.  
> degrees를 5° 이하로 제한하여 실제 위치 오차를 최소화합니다.

### 8.3 클래스 불균형 해소 (`_build_class_balanced_weights`)

1. 각 이미지에 포함된 전역 클래스 집합을 COCO annotation에서 추출
2. 클래스별 등장 이미지 수 계산 (`class_counts`)
3. 이미지 weight = 해당 이미지 클래스들의 `1/count` 평균
4. `WeightedRandomSampler`로 희귀 클래스 이미지 오버샘플링

### 8.4 배치 처리 (`collate_fn`)

- `None` 샘플(이미지 로드 실패 또는 유효 annotation 없음)을 자동으로 필터링
- 빈 배치 발생 시 `(None, None)` 반환 → 학습 루프에서 skip

---

## 9. 평가 지표

### 9.1 표준 COCO 지표

| 지표 | 설명 |
|------|------|
| `mAP@0.5` | IoU 임계값 0.5에서의 평균 정밀도 |
| `mAP@0.5:0.95` | IoU 0.5~0.95 구간 평균 (COCO 표준) |

### 9.2 Per-class AP (`_coco_per_class_breakdown`)

| 지표 | 설명 |
|------|------|
| `AP50` | 클래스별 mAP@0.5 |
| `AP50:95` | 클래스별 mAP@0.5:0.95 |
| `nGT` | 해당 클래스의 GT annotation 수 |

### 9.3 Per-class 커스텀 탐지 지표 (`PerClassAccumulator`)

| 지표 | 설명 |
|------|------|
| `accuracy_img` | 이미지 단위 존재/부재 정확도 (TP+TN)/(전체) |
| `precision` | TP/(TP+FP), IoU≥0.5 그리디 매칭 기반 |
| `recall` | TP/(TP+FN), IoU≥0.5 그리디 매칭 기반 |
| `mean_IoU_TP` | 매칭된 TP의 IoU 평균값 |
| `R²_cx, R²_cy` | TP 박스 중심점 x,y 회귀 적합도 |
| `R²_w, R²_h` | TP 박스 너비/높이 회귀 적합도 |
| `TP, FP, FN, TN` | 원시 탐지 카운트 |
| `img_TP/FP/TN/FN` | 이미지 단위 카운트 |

> **score_thr = 0.5** (`PERCLASS_SCORE_THR`): precision 안정성을 위해 저신뢰도 예측 필터링

---

## 10. 실험 추적 (TensorBoard & W&B)

### 10.1 TensorBoard

```bash
tensorboard --logdir C:/Users/<user>/.spyder-py3/runs/hazard_detection
```

| 로그 항목 | 주기 |
|-----------|------|
| `Loss/train` | 100 iteration마다 |
| `LR` | 100 iteration마다 |
| `mAP/<ds>_0.5` | 에폭마다 |
| `mAP/<ds>_0.5:0.95` | 에폭마다 |
| Per-class worst10 텍스트 | 에폭마다 |

### 10.2 Weights & Biases

```python
WANDB_PROJECT = "hazard-detection"
WANDB_ENTITY  = None        # 팀 사용 시 "your-team" 입력
WANDB_MODE    = "online"    # 오프라인 환경: "offline"
```

| 로그 항목 | 형태 |
|-----------|------|
| Loss, LR, mAP | Scalar |
| Per-class AP table | `wandb.Table` (`per_class_ap/<ds>`) |
| Per-class 탐지 지표 | `wandb.Table` (`per_class_det/<ds>`) |
| Best Model | W&B Artifact (`best_model`) |
| 체크포인트 | W&B Artifact (`checkpoint`) |
| 모델 파라미터/그래디언트 | `wandb.watch(log="all", log_freq=1000)` |

---

## 11. 체크포인트 및 Resume

### 11.1 체크포인트 저장 내용

```python
{
    "epoch":               int,   # 완료된 에폭 번호
    "iteration":           int,   # 누적 iteration 수
    "model_state_dict":    dict,  # 모델 가중치
    "optimizer_state_dict": dict, # 옵티마이저 상태
    "scheduler_state_dict": dict, # 스케줄러 상태
    "best_map":            float, # 지금까지의 최고 mAP@0.5
}
```

### 11.2 원자적 저장 방식

```
checkpoint.pth.tmp 로 임시 저장
    → os.replace() 로 checkpoint.pth 로 교체
```

파일 저장 도중 프로세스 종료 시 기존 체크포인트가 손상되지 않습니다.

### 11.3 Resume 설정

```python
RESUME = True   # False로 변경 시 처음부터 학습
```

Resume 시 `numpy._core.multiarray.scalar`를 안전 허용 목록에 추가하여
`weights_only=False` 로드 중 발생하는 역직렬화 오류를 방지합니다.

---

## 12. 실행 방법

### 12.1 최초 학습

```python
# train_eachclass.py 상단 설정
RESUME = False
NUM_EPOCHS = 40
```

```bash
python train_eachclass.py
```

### 12.2 이어 학습 (Resume)

```python
RESUME = True  # 기본값
```

```bash
python train_eachclass.py
# checkpoint.pth 자동 감지 후 이어 학습
```

### 12.3 학습 없이 테스트만 실행

`start_training()` 내의 학습 루프를 건너뛰고 테스트 블록만 실행하려면,
`NUM_EPOCHS`를 `start_epoch`보다 작게 설정하면 됩니다.

```python
# checkpoint에서 start_epoch=30 로 resume된 경우
NUM_EPOCHS = 30  # 학습 루프 skip → 테스트만 수행
```

### 12.4 오프라인 환경 (W&B)

```python
WANDB_MODE = "offline"
```

추후 온라인 동기화:
```bash
wandb sync C:/Users/<user>/.spyder-py3/runs/hazard_detection/<run_name>/wandb/offline-run-*/
```

---

## 13. 주요 클래스 및 함수 레퍼런스

### 13.1 데이터 관련

| 이름 | 유형 | 역할 |
|------|------|------|
| `XrayHazardDataset` | Class | COCO JSON 기반 X-ray 데이터셋, 전역 ID 매핑 |
| `get_transform(train)` | Function | ToTensor + (학습 시) 플립·회전 변환 반환 |
| `RandomRotateSmall` | Class | ±degrees 범위 소규모 회전 (bbox 미변환) |
| `collate_fn(batch)` | Function | None 샘플 필터링 후 배치 반환 |
| `_build_class_balanced_weights` | Function | 클래스 불균형 보정 WeightedRandomSampler 가중치 계산 |

### 13.2 평가 관련

| 이름 | 유형 | 역할 |
|------|------|------|
| `PerClassAccumulator` | Class | 클래스별 precision/recall/IoU/R² 누적 계산 |
| `evaluate_single_dataset` | Function | 단일 데이터셋에 대한 COCO + 커스텀 지표 평가 |
| `_match_per_class` | Function | Score 내림차순 그리디 매칭 (GT:Pred = 1:1) |
| `_box_iou_np` | Function | NumPy 기반 배치 IoU 계산 [Na, Nb] |
| `_xyxy_to_cxcywh` | Function | xyxy → 중심점+너비높이 형식 변환 |
| `_r2_score` | Function | R² (결정계수) 계산 (다차원 지원) |
| `_coco_per_class_breakdown` | Function | COCOeval 결과에서 클래스별 AP50/AP50:95 추출 |

### 13.3 로깅 관련

| 이름 | 유형 | 역할 |
|------|------|------|
| `_log_det_metrics_table` | Function | Per-class 탐지 지표를 W&B Table + TB 텍스트로 기록 |
| `_log_per_class_ap_to_wandb` | Function | Per-class AP를 W&B Table로 기록 |

### 13.4 모델 관련

| 이름 | 유형 | 역할 |
|------|------|------|
| `create_model(num_classes)` | Function | COCO 사전학습 Faster R-CNN 생성 + Head 교체 |
| `set_seed(seed)` | Function | 전역 시드 고정 (random/numpy/torch/cudnn) |

---

## 14. 트러블슈팅

### 14.1 CUDA OOM (메모리 부족)

```python
# 배치 크기 감소
BATCH_SIZE = 8  # 16 → 8

# 또는 멀티 스케일 축소
model.transform.min_size = [600, 800]  # 1000 제거
```

### 14.2 Resume 실패

```
Resume failed: ... Starting fresh.
```

체크포인트 파일이 손상된 경우 자동으로 처음부터 학습을 시작합니다.  
손상된 `.pth` 파일 삭제 후 재실행하거나, `RESUME = False`로 설정합니다.

### 14.3 W&B 연결 오류

```python
WANDB_MODE = "offline"  # 내부망/폐쇄망 환경
```

### 14.4 `torch.isfinite(losses)` 조건에서 배치 skip

손실이 `inf` 또는 `nan`인 배치는 자동으로 건너뜁니다.  
이 현상이 빈번하면 `BASE_LR`을 낮추거나 gradient clipping 활성화를 고려하세요:

```python
# start_training() 내 주석 해제
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### 14.5 annotation 파일 없음 경고

```
Warning: missing <path>/<file>.json
```

해당 데이터셋은 자동으로 건너뜁니다. 경로 오류 여부를 확인하세요.

### 14.6 데이터셋 카테고리 없음 오류

```
ValueError: No categories found.
```

6개의 COCO JSON 파일 모두 `"categories"` 필드가 없거나 파일이 존재하지 않을 때 발생합니다.  
JSON 파일 경로와 내용을 확인하세요.

---

## 15. 향후 개선 방향

| 항목 | 내용 |
|------|------|
| **Gradient Clipping** | `max_norm=10.0` 주석 해제로 학습 안정성 향상 |
| **W&B Sweep** | `BASE_LR`, `WEIGHT_DECAY`, `BATCH_SIZE` 자동 하이퍼파라미터 탐색 |
| **num_workers 증가** | 현재 `num_workers=0` → 멀티프로세스 로딩으로 I/O 병목 해소 |
| **bbox 회전 보정** | `RandomRotateSmall`에 bbox 좌표 회전 변환 추가 |
| **EfficientDet / DINO** | 더 높은 정확도를 위한 최신 객체 탐지 모델로 교체 |
| **Test Time Augmentation (TTA)** | 평가 시 멀티 스케일 앙상블로 mAP 향상 |
| **ONNX 내보내기** | 배포를 위한 모델 경량화 및 추론 최적화 |
| **같은 run으로 W&B 이어쓰기** | `id=RUN_NAME, resume="allow"` 주석 해제 |

---

## 참고

- [Torchvision Faster R-CNN 공식 문서](https://pytorch.org/vision/stable/models/faster_rcnn.html)
- [COCO Dataset 형식 가이드](https://cocodataset.org/#format-data)
- [Weights & Biases 빠른 시작](https://docs.wandb.ai/quickstart)
- [pycocotools 사용법](https://github.com/cocodataset/cocoapi)

---

> **작성 기준**: `train_eachclass.py` 소스코드 직접 분석 기반  
> **최종 업데이트**: 2025년
