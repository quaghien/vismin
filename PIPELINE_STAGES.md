# VisMin Pipeline - Chi Tiết Các Stage

## Tổng Quan

VisMin pipeline tạo minimal-change image-text pairs cho 3 loại dataset:
1. **COCO/VSR** (Real images - Object/Attribute editing)
2. **Relation** (Synthetic images - Spatial relations)
3. **Counting** (Synthetic images - Object counting)

---

## Models Được Sử Dụng Thực Tế

### 1. GLIGEN (Layout-to-Image Diffusion)
- **Input:** Prompt + bounding boxes + background prompt
- **Output:** Ảnh 512x512 được generate từ layout
- **Dùng cho:** Relation và Counting datasets
- **Ý nghĩa:** Tạo ảnh synthetic từ layout định sẵn (vị trí các objects)
- **Diffusion steps:** 50 iterations để từ noise → ảnh rõ nét

### 2. SAM (Segment Anything Model)
- **Input:** Generated image + bounding box của từng object
- **Output:** Segmentation masks (3 masks/object, chọn mask tốt nhất)
- **Dùng cho:** Relation và Counting datasets
- **Ý nghĩa:** Verify objects có xuất hiện đúng vị trí không bằng segmentation
- **Metrics:** Confidence score (0.96-0.99) + IoU với bounding box
- **Chi tiết:** SAM chạy 50 steps cho MỖI object → "4/5 objects completed" nghĩa là đã generate masks cho 4/5 objects

### 3. LLaVA-v1.6-Mistral-7B (VQA Model)
- **Input:** Generated/edited image + câu hỏi VQA
- **Output:** Câu trả lời + TIFA score (0-1)
- **Dùng cho:** COCO/VSR datasets only (KHÔNG dùng cho Relation/Counting)
- **Ý nghĩa:** Verify ảnh có chứa đúng objects/attributes không
- **SBERT fallback:** Khi LLaVA trả lời không khớp choices, dùng SBERT tìm answer gần nhất

### 4. Stable Diffusion XL Inpainting
- **Input:** Original image + mask + edited prompt
- **Output:** Edited image (inpaint vùng masked)
- **Dùng cho:** COCO/VSR datasets (edit real images)
- **Fallback:** Thay thế LAMA inpainting khi LAMA unavailable
- **Ý nghĩa:** Chỉnh sửa vùng cụ thể trong ảnh, giữ nguyên background

### 5. Grounding-DINO
- **Input:** Original image + text phrase (ví dụ: "glass of water")
- **Output:** Bounding boxes và masks của object
- **Dùng cho:** COCO/VSR datasets (tự động detect object cần edit)
- **Ý nghĩa:** Không cần manual annotation, tự động tìm object trong ảnh

### 6. OpenHermes-2.5-Mistral-7B (Language Model)
- **Input:** Prompts templates + captions/descriptions
- **Output:** Edit instructions JSON / VQA questions / Magic prompts
- **Dùng cho:** Tất cả datasets (Stage 1 và 2.5)
- **Ý nghĩa:** Generate metadata, instructions và questions cho pipeline

---

## 1. COCO/VSR Dataset (Real Images - Object/Attribute Editing)

**Đặc điểm:**
- Edit từ ảnh THẬT (COCO/VSR datasets)
- Thay đổi 1 object hoặc attribute (minimal change)
- Cần VQA filtering để đảm bảo chất lượng

### Stage 1: Generate Edit Instructions
**Models:** OpenHermes-2.5-Mistral-7B

**Input:** Caption gốc: "A glass of ice water sitting next to a wine glass"

**Output:**
```json
{
  "InputCaption": "A glass of ice water...",
  "SelectedPhrase": "glass of ice water",
  "EditedPhrase": "glass of milk",
  "EditedCaption": "A glass of milk sitting next to a wine glass",
  "Category": "object"
}
```

**Ý nghĩa:** LLM tạo minimal edits để tạo hard negatives

---

### Stage 2.5: Generate VQA Questions (BẮT BUỘC)
**Models:** OpenHermes-2.5-Mistral-7B

**Output:** `qa_annotations_{dataset}_{split}.json`

**Ý nghĩa:** Tạo questions để LLaVA verify ảnh edited đúng không

---

### Stage 3: Diffusion-guided Image Editing
**Models:** Grounding-DINO → SDXL Inpainting → LLaVA VQA

**Flow:**
```
Ảnh gốc → Grounding-DINO (mask) → SDXL Inpainting (edit) → 
Generated ảnh → LLaVA (verify) → TIFA score → Filter/Save
```

**Output:** `VIM_DATA_DIR/{dataset}_sdxl_edited_{split}/`

---

## 2. Relation Dataset (Synthetic - Spatial Relations)

**Đặc điểm:**
- Generate ảnh SYNTHETIC từ layout (không có ảnh gốc)
- Test spatial relations: above/below, left/right
- KHÔNG cần VQA filtering

### Stage 1: Generate Layout Instructions
**Output:**
```json
{
  "prompt": "A rabbit is below a flower",
  "bounding_boxes": [
    ["a rabbit", [180, 220, 152, 80]],
    ["a flower", [180, 50, 152, 50]]
  ],
  "background_prompt": "A garden"
}
```

### Stage 3: Layout-to-Image Generation
**Models:** SAM → GLIGEN

**SAM (50 steps × số objects):**
- Generate segmentation mask cho MỖI object
- "4/5 objects completed" = đã tạo xong masks cho 4/5 objects
- Mỗi object: 3 candidate masks → chọn best (highest confidence + IoU)

**GLIGEN (50 diffusion steps):**
- Time 0-25: Frozen layout (đảm bảo spatial relations)
- Time 26-50: Semantic refinement
- Loss optimization: cross-attention align với bounding boxes

**Output:** `VIM_DATA_DIR/generated_output_image/bootstrap_layout_relation_train/`

---

## 3. Counting Dataset (Synthetic - Object Counting)

**Đặc điểm:**
- Generate ảnh SYNTHETIC với nhiều objects giống nhau
- Test counting: 2-10 objects cùng loại

### Stage 1: Generate Layout Instructions
**Output:**
```json
{
  "prompt": "A scene with five eggs",
  "bounding_boxes": [
    ["an egg", [20, 200, 40, 60]],
    ["an egg", [80, 200, 40, 60]],
    ["an egg", [140, 200, 40, 60]],
    ["an egg", [200, 200, 40, 60]],
    ["an egg", [260, 200, 40, 60]]
  ]
}
```

### Stage 3: Layout-to-Image Generation
**Models:** SAM (5 objects × 50 steps) → GLIGEN

**Output:** `VIM_DATA_DIR/generated_output_image/bootstrap_layout_counting_train/`

---

## So Sánh 3 Loại Dataset

| Feature | COCO/VSR | Relation | Counting |
|---------|----------|----------|----------|
| **Ảnh nguồn** | Real images | Synthetic | Synthetic |
| **Method** | Inpainting/Edit | Layout-to-image | Layout-to-image |
| **Stage 2.5** | ✅ VQA questions | ❌ Skip | ❌ Skip |
| **Models** | DINO + SDXL + LLaVA | SAM + GLIGEN | SAM + GLIGEN |
| **VQA Filtering** | ✅ Required | ❌ Skip | ❌ Skip |
| **Acceptance** | Score-based | 100% | 100% |

---

## Memory Optimization

**Sau optimization:**
- COCO/VSR: Load LLaVA + SBERT
- Relation/Counting: Skip VQA models
- **Tiết kiệm:** ~15GB memory + ~7s loading time

---

## Giải Thích Logs

### SAM Mask Generation
```
100%|████| 50/50 [00:03<00:00, 14.95it/s]  ← SAM cho object 1
mask_sizes: [58 53 48], scores: [58 53 48]
Selected a mask with confidence: 0.96484375
```
**Nghĩa là:** SAM tạo 3 masks khác nhau (58, 53, 48 pixels), chọn mask đầu (58 pixels) vì confidence cao nhất

### GLIGEN Diffusion
```
time index 0, loss: 10.028, iteration: 1
time index 29, loss: 8.492, iteration: 1
```
**Nghĩa là:** Loss giảm từ 10.0 → 8.5 qua 30 steps = diffusion converge thành công
