# PRET is a few-shot system for pan-cancer recognition without example training


## Introduction

**PRET** (**P**an-cancer **R**ecognition without **E**xample **T**raining) is an innovative approach for multi-cancer diagnostics and tasks that eliminates task-specific model training. Utilizing a few labeled examples, PRET empowers pathological foundation models with the capability to directly recognize pan-cancer in the manner of in-context learning (ICL) to learn from the inference stage. PRET fully accounts for the unique characteristics of whole slide images, where massive patch tiles preserve rich local information, thereby facilitating exceptional recognition capabilities. By offering a flexible and cost-effective solution for pan-cancer recognition, PRET paves the way for accessible and equitable AI-based pathology systems, particularly benefiting minority populations and underserved regions.

* This is the data flow of our method to assist in understanding the code.

![](https://github.com/xmed-lab/PRET/blob/main/preview.png)

* This repo is originally relased at ([HF Code](https://huggingface.co/yili7eli/PRET)).

* Besides the code, all in-house datasets and labels are available too ([HF DATA](https://huggingface.co/datasets/yili7eli/PRET/tree/main)).


## Download Resources

* Download this repo for the code:
```
git clone git@github.com:xmed-lab/PRET.git
```

* Download our datasets and labels in PRET/data
```
cd PRET
huggingface-cli download yili7eli/PRET --local-dir data --repo-type dataset
```

* The default foundation model is provided in this repo. If other foundation models are needed, please download their extra model, code, and dependencies following their websites. Their feature extraction pipelines are included in the core/feature_extractor.py.

* Download public datasets from their official websites and save slides in folders like "data/NSCLC/images".


## Install
* Install Python packages via pip:
```
cd PRET
pip install -r requirements.txt
```

* Install libvips via apt (Ubuntu) for WSI slicing:
```
sudo apt install libvips-dev
```


## Dataset Process
Slice WSIs to patches; process annotations; generate data information (given in data_info/).
* For the in-house dataset and the TCGA datasets:
```
#bash scripts/prepare_dataset.sh [DATASET_NAME]; for example:
bash scripts/prepare_dataset.sh ESCC
```
* For CAMELYON datasets:
```
#bash scripts/prepare_dataset_camelyon.sh [DATASET_NAME]; for example:
bash scripts/prepare_dataset_camelyon.sh CAMELYON16
```


## Batch Evaluation
Run scripts/batch_run.py for convenient batch evaluation after dataset processing.

* The batch evaluation script runs all involved tasks, prompt types, and repeat experiments for a dataset.
* The script runs multiple experiments at once; the number of repeated experiments is n_tasks x n_prompts x n_repeats.

```
#python scripts/batch_run.py [DATASET_NAME] [MODEL_NAME_OR_WEIGHTS] [PARALLEL_TASK_NUM]
python scripts/batch_run.py ESCC model.pth 4
```

## Single Evaluation
Run scripts/run.py for a single benchmark.

* Run a single benchmark by assigned task, prompt.
* The number of repeated experiments is n_repeats only.
* The modes include default, baselines, and eval (without hyperparameter search).
```
#python scripts/run.py [GPU_ID] [DATASET_NAME] [TASK] [MODE] [PROMPT_TYPE] [MODEL_NAME_OR_WEIGHTS]
python scripts/run.py 0 ESCC screening default slideLabel model.pth
```

## H5 Feature Evaluation
This optimized fork can run directly from pre-extracted WSI patch features saved as `.h5` or `.hdf5` files. Each h5 file is treated as one slide and must contain a `features` key. A `coordinates` key is optional:

* `features`: a 2D array with shape `(num_patches, feature_dim)`.
* `coords` or `coordinates`: optional 2D array with shape `(num_patches, 2)` or `(num_patches, >=2)`. The first two columns may be patch grid coordinates `(x, y)` such as `0,1,2...`, or level-0 pixel top-left coordinates such as `0,512,1024...`. `H5_COORDINATE_MODE=auto` detects this from the coordinate step; step values below `H5_PIXEL_STEP_THRESHOLD` are treated as patch-grid coordinates, and larger steps are treated as pixel coordinates. If this key is missing, PRET generates deterministic row-major synthetic grid coordinates so slide-level h5 evaluation can continue.

Put all h5 files in one folder, for example:

```
data/MY_H5/h5/
  slide_001.h5
  slide_002.h5
  slide_003.h5
```

If you have slide-level labels, create a `data_info/MY_H5.json` file whose keys match the h5 file names without extension:

```json
{
    "slide_001": {
        "wsi_label": 0,
        "fixed_test_set": false
    },
    "slide_002": {
        "wsi_label": 1,
        "fixed_test_set": false
    }
}
```

For binary screening, use labels `0` and `1` and set `--class_num 1`. For multi-class subtyping, use labels `1..N` and set `--class_num N`. For example, a 7-class task should use labels `1, 2, 3, 4, 5, 6, 7` and `--class_num 7`.

Then run PRET directly on the h5 features with the reusable shell wrapper:

```
DATASET_NAME=MY_H5 \
H5_DIR=data/MY_H5/h5 \
DATASET_INFO=data_info/MY_H5.json \
CLASS_NUM=1 \
EXAMPLE_NUM=1 \
VAL_NUM=6 \
TEST_NUM=6 \
bash scripts/run_h5_eval.sh
```

The wrapper forwards its environment variables to `core/main.py`. A direct command is equivalent:

```
python core/main.py \
  --mode eval \
  --topk 3 \
  --temperature 10 \
  --related_thresh 0.8 \
  --example_num 1 \
  --raw_feature_path data/MY_H5/h5 \
  --wsi_path data/MY_H5/images \
  --dump_features data/MY_H5/collected_features \
  --dataset_info data_info/MY_H5.json \
  --seed 1024 \
  --top_instance 3 \
  --test_num 6 \
  --val_num 6 \
  --prompt_type slideLabel \
  --prompt_path data/MY_H5/anno \
  --ignore 0 \
  --file_min_size 0 \
  --class_num 1 \
  --runs 1 \
  --dump_records records/MY_H5_screening_slideLabel_eval.npy
```

`--wsi_path` and `--prompt_path` can point to non-existing folders when you only evaluate slide-level h5 features without heatmap visualization or segmentation. If `data_info/MY_H5.json` is missing, or if a slide has no `wsi_label`, PRET assigns deterministic pseudo labels by h5 file order so the pipeline can be smoke-tested. These pseudo-label results are only for verifying that the code runs; they are not meaningful benchmark metrics.

### H5 mask evaluation from CSV annotations

For `prompt_type=mask` with pre-extracted h5 features, PRET can align patch labels directly to the h5 feature order. This is useful when each h5 file stores:

* `features`: a 2D array with shape `(num_patches, feature_dim)`.
* `coords` or `coordinates`: a 2D array with shape `(num_patches, >=2)`.

H5 `coordinates` can be stored in two formats. If they are level-0 pixel top-left coordinates, use `--h5-coordinate-mode pixel`, or leave `auto` when the coordinate step is large, for example `256` or `512`. If they are patch-grid coordinates, for example step `1`, use `--h5-coordinate-mode grid`, or leave `auto` and the converter will detect grid mode. In grid mode, `--patch-scale` means the level-0 pixel size of one patch; when `--patch-scale 0`, the converter tries to infer it from `--size-json` or `--wsi-dir`, then falls back to `512`. If the h5 only keeps a tissue subset instead of the full slide grid, pass the real patch size explicitly with `--patch-scale 256` or `--patch-scale 512`.

The CSV annotation file must have three columns:

```
wsi_path,labels,coordinates
```

`wsi_path` is the WSI path, `labels` is mapped by a JSON label map, and `coordinates` is a flat list of normalized coordinates in `[0, 1]`: `x1,y1,x2,y2,...`. URL-escaped slide names are decoded before matching h5 files. For example, `2026-004317%238%231.sdpc` is converted to `2026-004317#8#1` and then matched with `2026-004317#8#1.h5`; if the CSV already contains `#`, the name is kept unchanged. The `labels` field may be a plain label or a JSON/Python-style list such as `["fibroadenoma"]`.

Two points are interpreted as opposite rectangle corners. More than two points are interpreted as a polygon. For h5-only workflows, use `prompt_type=mask` after converting these rectangles/polygons to h5-aligned patch labels; PRET's `box` and `roughMask` prompt modes are XML/WSI weak-prompt modes and are less suitable when only h5 features are available. If `--h5-dir` is provided and its coordinates are pixel coordinates, the converter can infer each slide size as `max_coordinate + inferred_patch_size`. If h5 coordinates are patch-grid coordinates, pass `--size-json`, `--wsi-dir`, or explicit `--patch-scale` so normalized CSV annotations can be mapped back to level-0 pixels correctly. You can still provide a size JSON to override inferred sizes:

```json
{
    "slide_001": {"width": 120000, "height": 80000},
    "slide_002": {"width": 100000, "height": 90000}
}
```

For SDPC slides, the converter tries OpenSlide first and then the optional `opensdpc` reader when `--slide-reader auto` is used. If OpenSlide cannot read your `.sdpc` files, install the SDPC reader in the same Python environment used for conversion:

```
pip install opensdpc
```

To create only the `size_json` file from a folder of `.sdpc` slides:

```
python prepare/wsi_dir_to_size_json.py \
  --wsi-dir /path/to/sdpc_folder \
  --out data_info/MY_H5_size.json \
  --slide-reader opensdpc
```

The same operation is also available from the main converter:

```
python prepare/csv_to_pret_annotations.py \
  --wsi-dir /path/to/sdpc_folder \
  --size-json-out data_info/MY_H5_size.json \
  --write-size-json-only \
  --slide-reader opensdpc
```

The JSON keys are URL-decoded slide stems by default, so `2026-004317%238%231.sdpc` becomes `2026-004317#8#1`. If your SDPC files are nested in subfolders, add `--recursive` for `wsi_dir_to_size_json.py` or `--wsi-recursive` for `csv_to_pret_annotations.py`.

Label conversion rules:

* `--auto-label-map` assigns IDs by first appearance in the CSV: the first new tumor name becomes `1`, the next new tumor name becomes `2`, and so on. Use `--label-map-out` to save this mapping. If you already have a stable mapping, pass it with `--label-map` instead.
* For binary or ordinary single-label multiclass tasks, use `--wsi-label-mode binary`, `single-label`, or `max-label`. The generated `data_info` writes only `wsi_label`.
* For multi-label tasks, use `--wsi-label-mode multi-label`. The generated `data_info` writes only `wsi_labels`, for example `"wsi_labels": [1, 3, 5]`; it does not write `wsi_label`.
* With `--wsi-label-mode multi-label`, the h5 patch label file has shape `(num_patches, class_num)` and is multi-hot by class. For binary mode, the h5 patch label file remains a one-dimensional `0/1` array.
* When `--h5-label-out` is used, CSV rows whose slide name has no matching h5 file are skipped by default and reported as warnings. Add `--no-skip-missing-h5` if you prefer a hard failure. This avoids falling back to SDPC size readers for slides that cannot produce h5-aligned labels anyway.

First generate h5-aligned patch labels and a PRET `data_info` file:

```
python prepare/csv_to_pret_annotations.py \
  --csv /path/to/annotations.csv \
  --auto-label-map \
  --label-map-out data_info/MY_H5_label_map.json \
  --wsi-dir /path/to/sdpc_folder \
  --size-json-out data_info/MY_H5_size.json \
  --h5-dir data/MY_H5/h5 \
  --h5-label-out data/MY_H5/patch/h5_labels \
  --data-info-out data_info/MY_H5_mask.json \
  --h5-coordinate-mode auto \
  --patch-scale 0 \
  --prompt-type mask \
  --wsi-label-mode multi-label \
  --slide-reader auto
```

This writes:

* `data/MY_H5/patch/h5_labels/{slide}.npy`: one label array per slide, strictly aligned to the h5 `features` row order.
* `data_info/MY_H5_mask.json`: slide-level `wsi_labels`, `h5_patch_labels`, and `pos_patch_num`.
* `data_info/MY_H5_label_map.json`: label names assigned in first-seen CSV order as `1, 2, 3, ...`.
* `data_info/MY_H5_size.json`: slide level-0 sizes read from the SDPC folder and/or inferred from h5 coordinates.

Add `--mask-out data/MY_H5/patch/gt` if patch-grid PNGs are also needed for inspection or non-h5 workflows. Writing PNGs requires OpenCV (`cv2`), but h5-aligned `.npy` labels only require `h5py` and `numpy`.

By default, all non-zero label IDs are treated as positive regions. Use `--positive-labels tumor invasive` or `--positive-labels 1 2` if only selected labels should become positive mask regions. If unannotated negative slides are needed for a binary or single-label task, add them to the generated `data_info` JSON with `wsi_label: 0` and `fixed_test_set: false`. For multi-label tasks, a negative slide should use `wsi_labels: []`.

Then run h5 WSI-level multi-label evaluation. PRET fits one threshold per class on the validation split, combines the per-class decisions back into a multi-hot WSI prediction such as `[1, 0, 0, 0, 1, 0]`, and reports multi-label metrics:

```
DATASET_NAME=MY_H5 \
H5_DIR=data/MY_H5/h5 \
DATASET_INFO=data_info/MY_H5_mask.json \
DUMP_FEATURES=data/MY_H5/collected_features_mask \
PROMPT_TYPE=mask \
MULTILABEL=1 \
CLASS_NUM=6 \
H5_COORDINATE_MODE=auto \
H5_PIXEL_STEP_THRESHOLD=16 \
H5_PATCH_SIZE=0 \
EXAMPLE_NUM=8 \
VAL_NUM=100 \
TEST_NUM=-1 \
RUNS=5 \
bash scripts/run_h5_eval.sh
```

Set `CLASS_NUM` to the number of entries in the generated label map. Add `SEG=1` only when you want patch-level segmentation metrics instead of WSI-level multi-label classification metrics.

Use a fresh `DUMP_FEATURES` directory for mask runs, because PRET skips feature collection when a cached `dump_features/{slide}.npy` file already exists.

You can create a small fake h5-only dataset for a local smoke test:

```
bash scripts/run_fake_h5_binary.sh
```

You can also run a 7-class h5 smoke test:

```
bash scripts/run_fake_h5_7class.sh
```

For custom fake h5 data, control the number of generated classes. Add `--omit_coordinates` to smoke-test h5 inputs that only contain `features`:

```
python scripts/make_fake_h5_dataset.py --out data/FAKEH5_7CLASS/h5 --slides 70 --patches 96 --dim 64 --classes 7

DATASET_NAME=FAKEH5_7CLASS \
H5_DIR=data/FAKEH5_7CLASS/h5 \
DATASET_INFO=data_info/FAKEH5_7CLASS.json \
CLASS_NUM=7 \
EXAMPLE_NUM=1 \
VAL_NUM=21 \
TEST_NUM=21 \
bash scripts/run_h5_eval.sh
```


## Code Explanation
Code flow: a. scripts/batch_run.py -> b. core/feature_extractor.py -> c. scripts/run.py -> d. core/main.py -> e. core/modules.py


* a. scripts/batch_run.py: It first extracts the feature by invoking the core/feature_extractor.py as:
```python
# extract feature by invoking the core/feature_extractor.py (line 73)
if not os.path.exists(FEAT_DIR):
    command = 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, visible_devices)) + ' python -u -m torch.distributed.launch --nproc_per_node=' + str(len(visible_devices)) + ' --master_port=2333 core/feature_extractor.py ' + ' --data_path ' + PATCH_DIR + ' --pretrained_weights ' + model + ' --dump_features ' + FEAT_DIR + ' --num_workers 8 --batch_size_per_gpu 32 --arch ' + arch
os.system(command)
```

* a. scripts/batch_run.py: Then, it gets commands from core/feature_extractor.py for each task, prompt, and mode as:
```python
# invoke the scripts/run.py for the command for each benchmark (line 99)
commands.append(get_run_command(dataset, task, mode, prompt_type, model))
# execute commands using multiprocessing.Pool with GPU management (line 116)
pool.apply_async(execute_bash_command, args=(command, out_text), kwds={}, callback=None)
```

* b. core/feature_extractor.py: It first loads the model, supporting diverse foundation models:
```python
# We support many foundation models. Here, we just show two examples for reference.
# We have provided the weights of the default model; others need to download their code, dependencies, and weights following their official websites.
# load the default model from local weights
if "vit_small" in args.arch:
    sys.path.append("network")
    import vision_transformer as vits
    model = vits.__dict__[args.arch](patch_size=8, num_classes=0)
    utils.load_pretrained_weights(model, args.pretrained_weights)
    print(f"Model {args.arch} 8x8 built.")
# load from hugging face, you are required to clone the code in the home dir with its dependencies
elif args.arch == 'conch':
    sys.path.append(os.path.expanduser('~/CONCH'))
    from conch.open_clip_custom import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.pretrained_weights)
...
```

* b. core/feature_extractor.py: Then, it extracts features for patch tiles according to the inference codes of FMs:
```python
# two mode, patch feature or zero-shot features
if arch == 'conch':
    feats = model.encode_image(imgs, proj_contrast=False, normalize=False)
    #feats = model.encode_image(imgs, proj_contrast=True, normalize=False) # for zero-shot
# use its own function
elif arch == 'plip':
    feats = model.get_image_features(imgs)
# virchow needs feature concat
elif arch == 'virchow':
    feats = model(imgs)
    class_token = feats[:, 0]
    patch_tokens = feats[:, 1:]
    feats = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
# used for zero-shot only
elif arch == 'musk':
    feats = model(image=imgs.half(), with_head=True, out_norm=False)[0]
# other models, directly apply the forward function
else:
    feats = model(imgs)
```

* c. core/run.py: It contains basic information for benchmarks including folders, hyper-parameters, data split, with detailed comments as:
```python
# ===================== set data split ======================
# data split principle
# 1) TCGA 20% test similating 5-fold, 100 val or 50 val if not sufficient
# 2) New datasets is smaller than TCGA, thus 1/3 val 2/3 test for slides out of example
# 3) Cross race NSCLC, White follow TCGA split, others follow new datasets
```

* c. core/run.py: Then, it genetates the running command for core/main.py with args:
```
# ===================== generate the command ======================
# prepare args for the core/main.py
command = 'python -u core/main.py --mode ' + mode + ' --topk ' + str(TOPK) + ' --temperature ' + str(TEMP) + ' --related_thresh ' + str(THRESH) + ' --example_num ' + str(SHOT_NUM) + ' --raw_feature_path ' + FEAT_DIR + ' --wsi_path '  + WSI_DIR + ' --dump_features ' + COLLECTED_FEAT_DIR + ' --dataset_info data_info/' + dataset + '.json --seed ' + str(SEED) + ' --top_instance ' + str(TOP_INS) + ' --test_num ' + str(TEST_NUM) + ' --val_num ' + str(VAL_NUM) + ' --val_ratio ' + str(VAL_RATIO) + ' --prompt_type ' + prompt_type + ' --prompt_path ' + PROMPT_DIR + ' --ignore ' + str(IGNORE) + ' --multiple_num 1 2 4 8 --file_min_size ' + str(FILE_MIN_SIZE) + ' --c ' + str(CLS_NUM) + ' --ignore_query ' + str(IGNORE_QUERY) + ' --dump_records ' + RECORDS + '.npy'
if task == 'segmentation':
    command += ' --seg '
# save the records in both .txt and .npy formats
command += ' &> ' + RECORDS + '.txt'
```
* d. core/main.py: It is responsible for parsing parameters, collecting features, selecting models, and then evaluating PRET and baselines. The code function is "evaluation", where we have provide many comments to explain the code parts, see comments below:
```python
# ====================== repeat experimets n=args.runs ====================== (line 158)
# ====================== data split ====================== (line 163)
# ====================== run for each class ====================== (line 252)
# ====================== process example and prompts ====================== (line 257)
# ====================== apply in-context tagger ====================== (line 316)
# ====================== predict for test slides (queries)====================== (line 357)
# ====================== discriminative instance miner for subtyping ====================== (line 371)
# ====================== inference, including classifier, aggregator, post processer ====================== (line 379)
# ====================== process validation set and assign label ====================== (line 411)
# ====================== count and record results ====================== (line 468)
```

* e. core/modules: The core modules are implemented in this file, including the in-context tagger, classifier, miner, aggregator, and post processor. We allocat these modules with comments as follows:
```python
# ====================== instance miner for subtyping ====================== (line 226)
# ====================== basic tagger (algorithm 1) ====================== (line 251)
# ====================== in-context tagger (algorithm 2) ====================== (line 273)
# ====================== in-context tagger for subtyping (algorithm 2) ====================== (line 327)
# ====================== in-context classifier ====================== (line 399)
# ====================== attention aggregator ====================== (line 418)
# ====================== patch reshape to wsi for heatmap / seg ====================== (line 437)
# ====================== segmentation post processor ====================== (line 459)
```


## Record Examples
We have saved some record examples in the records folder to prove the reproducibility.
* The results of ESCC screening with slide label are recorded as follows:
```
eval 1-shot:
class:1 val auc:0.9919, test auc:0.9577, val acc: 0.9375, test f1: 0.8861, test acc: 0.8636
class:1 val auc:0.9762, test auc:0.9639, val acc: 0.9062, test f1: 0.8861, test acc: 0.8636
class:1 val auc:0.9028, test auc:0.9481, val acc: 0.8438, test f1: 0.9268, test acc: 0.9091
class:1 val auc:0.8988, test auc:0.9279, val acc: 0.8438, test f1: 0.8649, test acc: 0.8485
class:1 val auc:0.8785, test auc:0.899, val acc: 0.8125, test f1: 0.8605, test acc: 0.8182
auc mean: 0.9393, auc std: 0.0235, f1 mean: 0.8849, f1 std: 0.0235, acc mean: 0.8606, acc std: 0.0294
eval 2-shot:
class:1 val auc:0.9603, test auc:0.9563, val acc: 0.9375, test f1: 0.8916, test acc: 0.8594
class:1 val auc:0.9484, test auc:0.9823, val acc: 0.875, test f1: 0.9067, test acc: 0.8906
class:1 val auc:0.8696, test auc:0.933, val acc: 0.8438, test f1: 0.8101, test acc: 0.7656
class:1 val auc:0.9818, test auc:0.9375, val acc: 0.9375, test f1: 0.8986, test acc: 0.8906
class:1 val auc:0.9883, test auc:0.9643, val acc: 0.9688, test f1: 0.8889, test acc: 0.8594
auc mean: 0.9547, auc std: 0.018, f1 mean: 0.8792, f1 std: 0.0351, acc mean: 0.8531, acc std: 0.0459
eval 4-shot:
class:1 val auc:0.942, test auc:0.9393, val acc: 0.9333, test f1: 0.9048, test acc: 0.871
class:1 val auc:0.9689, test auc:0.9697, val acc: 0.9333, test f1: 0.8421, test acc: 0.8065
class:1 val auc:0.9911, test auc:0.969, val acc: 0.9333, test f1: 0.9136, test acc: 0.8871
class:1 val auc:0.97, test auc:0.9081, val acc: 0.9333, test f1: 0.8571, test acc: 0.8226
class:1 val auc:0.9861, test auc:0.9912, val acc: 0.9667, test f1: 0.925, test acc: 0.9032
auc mean: 0.9555, auc std: 0.0289, f1 mean: 0.8885, f1 std: 0.0327, acc mean: 0.8581, acc std: 0.0373
eval 8-shot:
class:1 val auc:0.9893, test auc:0.9388, val acc: 0.9643, test f1: 0.8788, test acc: 0.8571
class:1 val auc:1.0, test auc:0.9864, val acc: 0.9643, test f1: 0.9855, test acc: 0.9821
class:1 val auc:0.9875, test auc:0.9857, val acc: 0.9286, test f1: 0.9143, test acc: 0.8929
class:1 val auc:0.9883, test auc:0.9829, val acc: 0.9286, test f1: 0.9375, test acc: 0.9286
class:1 val auc:0.9883, test auc:0.9802, val acc: 0.9286, test f1: 0.8462, test acc: 0.7857
auc mean: 0.9748, auc std: 0.0181, f1 mean: 0.9124, f1 std: 0.048, acc mean: 0.8893, acc std: 0.0662
```
* More detailed records are saved in the .npy files, including the data split, logits, predictions, and all results.


## Other Notes

### Dataset Files
* The class.json contains the class names (e.g. {"Adenocarcinoma, NOS": 1, "Squamous cell carcinoma, NOS": 2}).
* The label.txt contains file names with slide labels (e.g. xxx.svs,1).
* The images and anno folders store WSIs and annotations (via software ImageScope), respectively.

### Script Explanation:
* The scripts/batch_run.py firstly invokes core/feature_extractor.py for feature extraction. Then it generates running commands for target datasets, methods, prompts, and tasks.
* The scripts/run.py generates a command to invoke the core/main.py to run a specific task with repeated experiments (e.g., ESCC-screening-imgLabel).
* The core/feature_extractor.py extracts patch features, we have implemented some other foundation models. Some models require saving their code in the home folder, and some models use huggingface to download.
* The core/modules.py contains the implementation of core modules, including tagger, miner, classifier, aggregator, pos pocessor.
* The core/main.py is the main code, involving our method and baselines to evaluate multiple shots and repeats.

### Extensive Datasets
* The CAMELYON16-C simulates scan corruptions using the same slides of CAMELYON16, uncommand "RandomDistortions" in the core/feature_extractor.py to activate it. Besides, copy the dataset and data_info file from CAEMLYON16 to CAEMLYON16C.
* We provide the label.cvs for CAMELYON17 about the data source and LNM size. External experiments in Fig.5 use micro and macro LNM as CAMELYON16.
* Experiments about external prompts needs to combine multiple datasets. You can create a new dataset (e.g., data/PTC_QP2GD) and merge their files. Besides, set the "fixed_test_set" (e.g., in data_info/PTC_QP2GD.json) to true for the external hospital.

### Reproducibility
* The data_info provides a fixed data list. We get a fixed data split with seed 1024 to ensure the same examples, val slides and test slides.
* The different package versions may slightly change the results (within an acceptable range), including scikit-learn, CUDA, torch, torchvision, cv2, pillow, etc., that related to evaluation and data loading.

### Time Cost
* Package installation usually takes less than an hour, depending on network speed.
* Dataset downloading takes hours or a few day,s depending on network speed (over 800GB for our in-house datasets).
* The process involves slide slicing, feature extraction, hyperparameter search, multiple shot settings, multiple prompts, and repeated experiments, which can take a few or dozens of hours depending on the data scale, CPU, GPU, and IO speed.


### Optimization


**主要优化/增强如下：**

1. **大数据内存优化**
   - [core/modules.py](/Users/chenbozhou/Desktop/PRET/core/modules.py:21) 新增 `PRET_EXAMPLE_CHUNK`、`PRET_QUERY_CHUNK`。
   - [compute_similarity](/Users/chenbozhou/Desktop/PRET/core/modules.py:112) 改成按 query/example chunk 流式计算 top-k similarity，避免一次性构造巨大的 `example x query` 相似度矩阵。
   - [inference](/Users/chenbozhou/Desktop/PRET/core/modules.py:423) 也改为流式计算正负样本 similarity，降低显存/内存峰值。
   - [execute_miner](/Users/chenbozhou/Desktop/PRET/core/modules.py:257) 复用新的 chunked similarity，不再手动拼接大矩阵。

2. **支持直接读取 `.h5/.hdf5` 特征**
   - [core/main.py](/Users/chenbozhou/Desktop/PRET/core/main.py:44) 新增 h5 文件发现逻辑。
   - [core/main.py](/Users/chenbozhou/Desktop/PRET/core/main.py:96) 可读取 h5 中的 `features`，并把每个 h5 当作一个 slide。
   - [feature_processor](/Users/chenbozhou/Desktop/PRET/core/main.py:149) 新增 h5 分支，会把 h5 特征转换成原 PRET 后续流程需要的 `.npy` 信息结构。

3. **兼容没有 `coordinates` 的 h5**
   - [core/main.py](/Users/chenbozhou/Desktop/PRET/core/main.py:88) 新增 synthetic coordinates。
   - 如果 h5 只有 `features`，没有 `coordinates`，会自动生成 row-major 坐标，保证 slide-level h5 流程能继续跑。

4. **h5-only 数据集可以没有真实 WSI/annotation**
   - [get_wsi_suffix](/Users/chenbozhou/Desktop/PRET/core/main.py:116) 和 [get_wsi_size](/Users/chenbozhou/Desktop/PRET/core/main.py:125) 允许 `wsi_path` 不存在。
   - h5-only 的 `slideLabel` 评估不再强依赖真实 WSI 文件、xml、mask 等。

5. **缺失 `data_info` 或 label 时可 smoke test**
   - [load_dataset_info](/Users/chenbozhou/Desktop/PRET/core/main.py:63) 如果 h5 输入没有完整 `data_info`，会按文件顺序生成 deterministic pseudo label。
   - 这主要用于 pipeline smoke test，不适合真实指标评估。

6. **slideLabel 逻辑更适合 h5-only**
   - [core/main.py](/Users/chenbozhou/Desktop/PRET/core/main.py:298) 对 binary `slideLabel`，正负 slide 都可作为 labeled examples。
   - 原版更偏向已有 patch/slide 数据组织，优化版更适合只有 slide-level h5 特征的场景。

7. **运行脚本增强**
   - 新增 [scripts/run_h5_eval.sh](/Users/chenbozhou/Desktop/PRET/scripts/run_h5_eval.sh:1)，封装 h5 特征评估，默认使用 `--prompt_type slideLabel`。
   - 新增 [scripts/run_fake_h5_binary.sh](/Users/chenbozhou/Desktop/PRET/scripts/run_fake_h5_binary.sh:1) 和 [scripts/run_fake_h5_7class.sh](/Users/chenbozhou/Desktop/PRET/scripts/run_fake_h5_7class.sh:1)。
   - [scripts/run.py](/Users/chenbozhou/Desktop/PRET/scripts/run.py:6) 支持额外传入 `class_num`，并统一使用 `--class_num`。

8. **多分类 h5 支持更明确**
   - [core/main.py](/Users/chenbozhou/Desktop/PRET/core/main.py:999) 增加 `--class_num` 作为 `--c` 的别名。
   - fake h5 生成脚本支持 `--classes N`，方便测 7 类等多分类任务。

9. **本地测试/假数据工具**
   - 新增 [scripts/make_fake_h5_dataset.py](/Users/chenbozhou/Desktop/PRET/scripts/make_fake_h5_dataset.py:1)。
   - 支持生成带 `coordinates` 或不带 `coordinates` 的 h5 假数据。
   - 仓库里还加入了 `data_info/FAKE500.json` 和一批 `data/FAKE500` 测试 fixture。

10. **健壮性细节**
   - 特征归一化加了 `1e-8` 防止除零。
   - similarity 归一化加了 `clamp_min(1e-8)`，避免全相同分数时出现 NaN。
   - 多处改用 `torch.from_numpy(...astype(np.float32, copy=False))`，减少不必要拷贝。
   - query 循环里显式 `del` 和 `torch.cuda.empty_cache()`，进一步控制显存峰值。

11. **多分类评估路径加速**
   - 原版在多分类评估时，会对每个类别单独走一遍 `tagger -> miner -> inference -> validation`，外层是 class-level `for` 循环，这部分很难完全改成“所有类别一步并行”，因为每个类别的 pseudo label、foreground miner、threshold search 都依赖该类别自己的 reference 和分数分布。
   - 优化版没有强行把 class loop 改成多进程/多线程去抢一张 GPU，而是优先把真正的热点变成 GPU 上的批量矩阵运算：
     - [`compute_similarity`](core/modules.py#L112) 维持 query/example chunk 流式计算，但把 chunk 内 top-k 和 reduce 保留在 GPU 上，不再每个 chunk 强制 `.cpu()`。
     - [`aggregate_query_logits`](core/modules.py#L147) 把原来 top-instance 一个个 patch 做 self-attention 聚合的循环，改成按 `ATTENTION_QUERY_CHUNK` 的小批量矩阵运算。
   - 这意味着“类别间逻辑仍是串行的”，但“每个类别内部最耗时的相似度和聚合”已经更接近 GPU 友好的并行实现，通常比直接开 CPU 并行更稳，也更不容易因为显存争抢反而变慢。

12. **提升 GPU 占用率与利用率**
   - 原版 GPU 利用率偏低的核心原因，是相似度计算虽然在 GPU 上做了乘法，但每个 chunk 立刻 `.cpu()`，导致频繁的 device sync 和 host-device 拷贝。
   - 优化版把中间 top-k scores 保留在 GPU 上，直到该阶段真正结束再做后续处理，减少了同步开销。
   - 默认 chunk 也从 `2048` 提高到 `4096`：
     - `PRET_EXAMPLE_CHUNK`
     - `PRET_QUERY_CHUNK`
     - `PRET_ATTENTION_QUERY_CHUNK`
   - 这几个参数都可以通过环境变量覆盖，用于按具体 GPU 显存容量调优。例如：

```bash
PRET_EXAMPLE_CHUNK=8192 PRET_QUERY_CHUNK=4096 PRET_ATTENTION_QUERY_CHUNK=1024 \
python core/main.py ...
```

   - 另外，终端现在会输出 `[timing]` 和 `[memory]`，便于直接观察 `io/query/miner/infer` 哪一段才是真正瓶颈，而不是只盯着 `nvidia-smi`。

13. **高质量 reference token 稀疏化**
   - reference token 的目标不是“随机删点”，而是“尽量保留最强、最有区分度、同时又不完全重复的 token”。
   - 旧版实验性实现里带有少量随机补位；当前版本已经改成**全确定性选择**，核心思路是一个 `importance-aware coreset`：
     - 先按 label (`0 / 1 / 255`) 给 token 分预算，避免某一类 token 被整体压得过狠。
     - 对每个 label 内的 token，计算 importance：
       - `own-centroid similarity`：token 和本类中心有多接近。
       - `margin to other centroids`：token 相对其他类中心有多可分。
     - 先保留一批最强的 `anchor tokens`。
     - 再用带 diversity 的贪心选择继续补点：已经选中的 token 越能覆盖当前分布，后续就越倾向补那些“同样重要但和已选 token 不那么相似”的点。
   - 相关实现见：
     - [`sparsify_reference_tokens`](core/main.py#L233)
     - [`select_high_quality_tokens`](core/main.py#L198)
     - [`select_hierarchical_tokens`](core/main.py#L276)
   - 开关参数：
     - `--reference_token_budget`：reference token 总预算；`0` 表示不稀疏化。
     - `--reference_sparsify_strategy`：可选 `auto / quality / legacy / hierarchical`。`hierarchical` 会在每个 label 内递归二分 token，形成一个轻量层次化聚类树，再从每个叶簇中选择兼顾 cluster centrality 和 importance 的代表 token。
     - `--reference_anchor_ratio`：预算中先分给 strongest anchors 的比例，默认 `0.25`。
   - 本地 smoke test 的结论是：功能上没有问题，但如果预算压得太狠，精度仍可能下降。因此默认仍建议 `--reference_token_budget 0`，或者先用较保守的预算，从保留 70%-85% token 开始试。

14. **混合特征维度兼容**
   - 某些多分类数据可能会混用不同 foundation model 导出的特征，例如一部分 slide 是 `768` 维，另一部分是 `1536` 维。原版会在 `np.concatenate(example_feats, 0)` 处直接报错。
   - 优化版会在 example/query 进入主流程前自动对齐到共享维度，并在终端打印 warning，说明哪些 slide/query 发生了 truncation/padding。
   - 这主要用于保证 pipeline 能跑通和方便定位数据源问题；如果追求严格 benchmark，一般仍建议统一上游特征维度。


15. **Training-free research extensions for stronger PRET variants**
   - `--similarity_aggregation mean|softmax|adaptive` keeps the original top-k mean as default, and adds softmax/adaptive top-k reducers that emphasize closer in-context references without training.
   - `--context_centering none|example|query|joint` adds SimpleShot-style feature centering before each query, useful for cross-scanner or cross-foundation-model shifts.
   - `--spatial_smooth_strength`, `--spatial_smooth_radius`, and `--spatial_feature_weight` add coordinate-aware neighborhood smoothing over patch logits before WSI aggregation, injecting WSI spatial coherence while preserving the original attention aggregator.
   - `--conformal_alpha` records validation-calibrated conformal prediction-set statistics such as coverage, singleton rate, and average set size. This does not change PRET predictions; it adds uncertainty reporting for deployment and paper analysis.
   - The h5 wrapper forwards these settings through environment variables, for example:

```bash
SIMILARITY_AGGREGATION=adaptive CONTEXT_CENTERING=joint SPATIAL_SMOOTH_STRENGTH=0.25 SPATIAL_SMOOTH_RADIUS=1 CONFORMAL_ALPHA=0.1 bash scripts/run_h5_eval.sh
```

## Citation

The paper is coming soon (accepted and waiting for publication).

## License
```
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```
