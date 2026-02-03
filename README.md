# VisMin: Visual Minimal-Change Understanding

## Table of Contents
- [Dataset](#dataset)
- [Minimal-Change Image-Text Dataset Creation](#minimal-change-image-text-dataset-creation)
  - [LLM-guided Edit Instructions Generation](#minimal-change-text-pair-generation)
  - [Diffusion-guided Image Synthesis](#minimal-change-image-generation)
  - [Edited Image Quality Verification using Local-Global VQA Approach](#edited-image-quality-verification-using-local-global-vqa-approach)
- [Setup](#setup)
- [Acknowledgements](#acknowledgements)


## Dataset
The training dataset has 64,392 samples, and the VisMin dataset has 2,084 samples. The dataset is stored in a JSON format. Each entry contains the image path, caption, and a list of negative examples. The negative examples consist of the edited image path and edited caption.
- **Training Data:** 64,392 samples from VSR and COCO 2017 training split.
- **Benchmark Data:** 2,084 samples from COCO 2017 validation split, human-verified.

🔥 Exciting News! 🔥 The VisMin benchmark dataset is now available 🎉 Check it out [here](https://huggingface.co/datasets/mair-lab/vismin-bench) 🤗 



Example of a dataset entry in the training dataset: 
```json
{
  "image_path": "/coco/images/train2017/000000234136.jpg",
  "caption": "Two men holding a brown and white dog in a van.",
  "negatives": [
    {
      "edited_image_path": "/edited/coco/234136/0.png",
      "edited_caption": "Three men holding a brown and white dog in a van.",
    }
  ]
}
```

## Setup
```bash
git clone <https://github.com/rabiulcste/vismin>
cd vismin
pip install -r requirements.txt
```


## Minimal-Change Image-Text Dataset Creation
### LLM-guided Edit Instructions Generation
We use LLM to generate edit instructions. There are two approaches to generate these instructions: one with captions, which suggests object attribute changes following the style of in-context demonstrations, and another for spatial and counting changes, where we prompt LLM with in-context demonstrations to create the appropriate edit instructions with layouts.

Example of an llm-generated edit instruction (object attribute category):
```json
  {
      "InputCaption": "A glass of ice water sitting next to a wine glass.",
      "SelectedPhrase": "glass of ice water",
      "EditedPhrase": "glass of milk",
      "EditedRegionPhrase": "A glass of milk",
      "EditedCaption": "A glass of milk sitting next to a wine glass.",
      "Category": "object"
  }
```
Example of an llm-generated edit instruction (spatial and counting category):
```json
 "A paint brush is to the left of a palette.": [
      "[('a paint brush', [50, 200, 100, 312]), ('a palette', [362, 150, 150, 362])]\nBackground prompt: A realistic scene\nNegative prompt:\nCategory: relation(left of)"
  ]
```

#### Setup COCO Dataset (required for object/attribute category)

```bash
# 1. Tạo thư mục COCO
mkdir -p COCO_DATA_DIR/coco/input_dataset
cd COCO_DATA_DIR/coco/input_dataset

# 2. Tải annotations (nhẹ, ~250MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

# 3. Tải train images (~18GB - MẤT THỜI GIAN)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

# 4. Tải validation images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# 5. Tải SAM checkpoint (~2.4GB) - cần cho Relation/Counting
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 5. Quay lại thư mục project
cd ~/hienhq/vismin
```

Cấu trúc thư mục sau khi tải:
```
COCO_DATA_DIR/coco/input_dataset/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── train2017/
│   └── 000000000009.jpg (118,287 images)
└── val2017/
    └── 000000000139.jpg (5,000 images)
```

#### Stage 1: Generate Edit Instructions

To run the script, from the directory containing `ctrl_edit/`, execute:
```bash
# 1. For object/attribute category - VSR dataset
python -m ctrl_edit.llm_agent.minchange_text_pairs_gen \
  --dataset vsr \
  --prompt_type edit_instructgen_from_caption \
  --language_model_name gemini-2.5-flash-lite

# 2. For object/attribute category - COCO dataset (chia thành 5 phần để chạy song song)
# Chạy 5 lệnh này trên 5 máy khác nhau, mỗi máy chạy 1 phần:
# --chunk_index 0: chạy phần 1/5 (samples 0-2000)
# --chunk_index 1: chạy phần 2/5 (samples 2000-4000)
# --chunk_index 2: chạy phần 3/5 (samples 4000-6000)
# --chunk_index 3: chạy phần 4/5 (samples 6000-8000)
# --chunk_index 4: chạy phần 5/5 (samples 8000-10000)
python -m ctrl_edit.llm_agent.minchange_text_pairs_gen \
  --dataset coco \
  --split train \
  --prompt_type edit_instructgen_from_caption \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --num_samples 50 \
  --chunk_index 0 \
  --total_chunks 2

# 3. For spatial relation category (tạo synthetic, không giới hạn)
# Không cần chunking vì tạo layout từ đầu, không phụ thuộc dataset có sẵn
python -m ctrl_edit.llm_agent.minchange_text_pairs_gen \
  --dataset relation \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B

# Hoặc giới hạn số samples (ví dụ 50 samples):
python -m ctrl_edit.llm_agent.minchange_text_pairs_gen \
  --dataset relation \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --num_samples 50

# 4. For counting category (tạo synthetic, không giới hạn)
python -m ctrl_edit.llm_agent.minchange_text_pairs_gen \
  --dataset counting \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --num_samples 50
```

#### Stage 2: Generate Magic Prompts

**Chức năng:** Đọc output JSON từ Stage 1 và tạo "magic prompts" (enhanced descriptions) cho diffusion model.

**Input:** Load JSON files từ Stage 1:
- **COCO/VSR:** `llm_edits_{dataset}/{model}/edit_instructgen_from_caption_{split}.json`
  - VD: `llm_edits_coco/openhermes25mistral7b/edit_instructgen_from_caption_train_chunked/chunk_0_20260131_200322.json`
- **Relation/Counting:** `llm_edits_{dataset}/{model}/edit_instructgen_with_layout_{split}.json`
  - VD: `llm_edits_relation/openhermes25mistral7b/edit_instructgen_with_layout_train_20260131_204359.json`

**Output:** Magic prompts JSON:
- **COCO/VSR:** `llm_edits_{dataset}/{model}/magic_prompt_chunk_{X}_timestamp.json` (later copied to `phrase_enhanced`)
- **Relation/Counting:** `llm_edits_{dataset}/{model}/magic_prompt_edit_instructgen_with_layout_train_timestamp.json`

**Lưu ý:** 
- Với COCO chunked: cần copy output về đúng tên file `phrase_enhanced` (xem Stage 3)
- Magic prompt được thêm vào object/phrase để cải thiện chất lượng diffusion

Generating magic prompt (to be appended with the e.g. object name) for better diffusion guidance of input prompt:
```bash
# Chỉ định chính xác file JSON từ Stage 1 (RECOMMENDED khi có nhiều file)

# 1. VSR - chỉ định file cụ thể
python -m ctrl_edit.llm_agent.magic_prompt_gen \
  --dataset vsr \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --input_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_vsr/openhermes25mistral7b/edit_instructgen_from_caption_train_20260131_200322.json

# 2. COCO - chỉ định file chunk cụ thể
python -m ctrl_edit.llm_agent.magic_prompt_gen \
  --dataset coco \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --input_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_coco/openhermes25mistral7b/edit_instructgen_from_caption_train_chunked/chunk_0_20260131_210853.json

# 3. For spatial relation category
python -m ctrl_edit.llm_agent.magic_prompt_gen \
  --dataset relation \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --input_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_relation/openhermes25mistral7b/edit_instructgen_with_layout_train_20260131_210300.json

# 4. For counting category
python -m ctrl_edit.llm_agent.magic_prompt_gen \
  --dataset counting \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --input_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_counting/openhermes25mistral7b/edit_instructgen_with_layout_train_20260131_210455.json
```

#### Stage 2.5: Generate VQA Questions for Image Quality Verification (Required for Stage 3)

> ⚠️ **CHỈ cần chạy cho COCO và VSR**. Relation/Counting KHÔNG cần stage này vì `diffusion_with_layout.py` không sử dụng QA annotations.

**Chức năng:** Tạo câu hỏi VQA (Visual Question Answering) để verify chất lượng ảnh sau khi generate.

**Input:** Edit instructions JSON từ Stage 1 (qua `--input_json` parameter)

**Output:** `qa_annotations_{dataset}_{split}.json`

```bash
# === CHỈ CẦN CHẠY CHO COCO VÀ VSR ===

# For COCO (chunked) - chỉ định file từ Stage 1
python -m ctrl_edit.llm_agent.auto_filter_question_gen \
  --dataset coco \
  --split train \
  --chunk_index 0 \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --batch_size 4 \
  --input_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_coco/openhermes25mistral7b/edit_instructgen_from_caption_train_chunked/chunk_0_20260131_210853.json

# For VSR - chỉ định file từ Stage 1
python -m ctrl_edit.llm_agent.auto_filter_question_gen \
  --dataset vsr \
  --split train \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --batch_size 4 \
  --input_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_vsr/openhermes25mistral7b/edit_instructgen_from_caption_train_20260131_200322.json

# ❌ KHÔNG CẦN chạy cho relation và counting
# diffusion_with_layout.py không load qa_annotations
```

### Diffusion-guided Image Synthesis

**⚠️ Stage 3: Diffusion Pipeline - Tạo/Sửa Ảnh**

We have two approaches to generate minimal-change images:
1. **Masking and Inpainting** (for COCO/VSR): Uses Grounding-DINO for automatic masking + Stable Diffusion Inpainting
2. **Layout-to-Image** (for relation/counting): Uses GLIGEN layout-guided generation from scratch

**Prerequisites:**
- ✅ Stage 1 & 2 completed (edit instructions + magic prompts JSON files ready)
- ✅ For COCO/VSR: Source images downloaded + Stage 2.5 QA annotations
- ✅ For relation/counting: Only need Stage 1 edit instructions (no source images needed)

**Commands:**

```bash
# === FOR SPATIAL RELATION CATEGORY ===
# Generate from scratch using GLIGEN layout-to-image
# Requires: edit_instructgen_with_layout_train_*.json from Stage 1
python -m ctrl_edit.diffusion_with_layout \
  --repeats 1 \
  --frozen_step_ratio 0.5 \
  --no-scale-boxes-default \
  --sdxl --sdxl-step-ratio 0.4 \
  --dataset relation \
  --split train \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --edit_instructions_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_relation/openhermes25mistral7b/edit_instructgen_with_layout_train_20260131_210300.json

# === FOR COUNTING CATEGORY ===
# Generate from scratch using GLIGEN layout-to-image
# Requires: edit_instructgen_with_layout_train_*.json from Stage 1
python -m ctrl_edit.diffusion_with_layout \
  --repeats 1 \
  --frozen_step_ratio 0.5 \
  --no-scale-boxes-default \
  --sdxl --sdxl-step-ratio 0.4 \
  --dataset counting \
  --split train \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --edit_instructions_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_counting/openhermes25mistral7b/edit_instructgen_with_layout_train_20260131_210455.json

# === FOR COCO DATASET (Object/Attribute) ===
# Uses masking + inpainting on existing images
# Requires: edit_instructgen_from_caption_*.json + magic_prompt_*.json + qa_annotations_*.json
python -m ctrl_edit.diffusion_with_mask \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --dataset coco \
  --split train \
  --chunk_index 0 \
  --diffusion_model_name sdxl \
  --edit_instructions_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_coco/openhermes25mistral7b/edit_instructgen_from_caption_train_chunked/chunk_0_20260131_210853.json \
  --magic_prompt_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_coco/openhermes25mistral7b/phrase_enhanced_train_chunked/chunk_0.json

# === FOR VSR DATASET (Object/Attribute) ===
# Uses masking + inpainting on existing images
# Requires: edit_instructgen_from_caption_*.json + magic_prompt_*.json + qa_annotations_*.json
python -m ctrl_edit.diffusion_with_mask \
  --language_model_name teknium/OpenHermes-2.5-Mistral-7B \
  --dataset vsr \
  --split train \
  --diffusion_model_name sdxl \
  --edit_instructions_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_vsr/openhermes25mistral7b/edit_instructgen_from_caption_train_20260131_200322.json \
  --magic_prompt_json SYNTH_DIFFUSE_DATA_DIR/generated_question_for_verification/prompt_resources/llm_edits_vsr/openhermes25mistral7b/phrase_enhanced_train.json
```

**Key Parameters:**
- `--edit_instructions_json`: **[REQUIRED]** Path to Stage 1 output JSON file
- `--magic_prompt_json`: **[For COCO/VSR only]** Path to Stage 2 magic prompts  
- `--repeats`: Number of generation attempts per prompt
- `--sdxl`: Enable SDXL refinement for better quality
- `--sdxl-step-ratio`: SDXL refinement strength (0.0-1.0)

**Output:**
- Generated images: `VIM_DATA_DIR/bootstrap_layout_{dataset}_{split}/{image_id}/`
- Metadata: `annotations.json` with scores and edit details

### Edited Image Quality Verification using Local-Global VQA Approach
Verify images through the vqa filter approach. first generate the local-global vqa questions and answers using llm following edit instruction.

```bash
# To create the local-global VQA questions and answers using LLM-generated edit instructions from one of the previous step:
python -m ctrl_edit.llm_agent.auto_filter_question_gen  --language_model_name <name_of_language_model>

# Automatically filter out bad edited images using the local-global VQA approach:
python -m ctrl_edit.filters.tifa_filter --dataset <dataset_name>
```



## Pipeline Flow Summary

| Dataset | Stage 1 | Stage 2 | Stage 2.5 | Stage 3 | VRAM |
|---------|---------|---------|-----------|---------|------|
| **Relation** | ✅ | ✅ (không dùng) | ❌ | GLIGEN + LAMA | ~20GB |
| **Counting** | ✅ | ✅ (không dùng) | ❌ | GLIGEN + LLaVA | >30GB |
| **COCO/VSR** | ✅ | ✅ (dùng) | ✅ | DINO + SDXL + LLaVA | ~25GB |

**Models trong Stage 3:**
| Model | Loại | Input | Output | Dùng cho |
|-------|------|-------|--------|----------|
| **GLIGEN** | Layout-to-Image Diffusion | Prompt + bounding boxes | Ảnh 512x512 từ layout | Relation, Counting |
| **SAM** | Segmentation | Ảnh + bbox | Masks verify object | Relation, Counting |
| **LAMA** | Inpainting | Ảnh + mask | Ảnh đã xóa object (tạo negative) | Relation |
| **LLaVA** | VQA (7B) | Ảnh crop + câu hỏi | "Is this a dog?" → Yes/No | Counting, COCO/VSR |
| **Grounding-DINO** | Object Detection | Ảnh + text phrase | Bounding boxes + masks | COCO/VSR |
| **SDXL Inpainting** | Image Editing | Ảnh gốc + mask + prompt | Ảnh đã edit vùng masked | COCO/VSR |

---

### Chi Tiết Từng Loại Dataset

#### 1. Relation Dataset (Spatial Relations)
```
Stage 1: LLM generate → {"prompt": "A cat is above a dog", "bboxes": [[cat, x,y,w,h], [dog, x,y,w,h]]}
Stage 3: GLIGEN (layout→image) → SAM (verify objects) → LAMA (xóa 1 object tạo negative)
```
**Output cuối:** Cặp ảnh (positive, negative) + caption
- Positive: "A cat is above a dog" + ảnh có cả cat và dog
- Negative: "A cat is above a dog" + ảnh chỉ có cat (dog bị xóa bằng LAMA)

#### 2. Counting Dataset (Object Counting)
```
Stage 1: LLM generate → {"prompt": "Five dogs", "bboxes": [[dog, x1,y1,w,h], [dog, x2,y2,w,h], ...]}
Stage 3: GLIGEN (layout→image) → SAM (segment) → LLaVA (crop từng bbox, hỏi "Is this a dog?")
```
**Output cuối:** Ảnh + caption + TIFA score (verify từng object)
- "Five dogs" + ảnh 5 con chó + score 0.73 (4/5 objects verified)

#### 3. COCO/VSR Dataset (Object/Attribute Editing)
```
Stage 1: LLM analyze caption → {"input": "glass of water", "edited": "glass of milk", "caption_edited": "..."}
Stage 2: LLM enhance → "glass of milk" → ["transparent, cold, white liquid, dairy"]
Stage 2.5: LLM generate VQA → [{"question": "Is there milk?", "answer": "yes"}, ...]
Stage 3: DINO (detect "water") → SDXL (inpaint "milk") → LLaVA (verify bằng VQA questions)
```
**Output cuối:** Cặp (ảnh gốc, ảnh edited) + (caption gốc, caption edited)
- Original: "A glass of water on table" + ảnh gốc
- Edited: "A glass of milk on table" + ảnh đã inpaint vùng water→milk

---

### Data Cuối Cùng Được Gì?

| Dataset | Positive Sample | Negative Sample | Mục đích |
|---------|-----------------|-----------------|----------|
| **Relation** | Ảnh có 2 objects đúng vị trí | Ảnh thiếu 1 object | Test spatial understanding |
| **Counting** | Ảnh có N objects | (chưa implement negative) | Test counting ability |
| **COCO/VSR** | Ảnh gốc + caption gốc | Ảnh edited + caption edited | Test fine-grained attribute |

**Format cuối:**
```json
{
  "image_path": "original.jpg",
  "caption": "A glass of water on table",
  "negatives": [{
    "edited_image_path": "edited.png",
    "edited_caption": "A glass of milk on table"
  }]
}
```
