# SciVQA
## Models used for the task (general VLLMs)
- BLIP2 - 2.7B
- LLAVA - 13B , LLAVA - 7B
- OpenFlamingo v2 - 7B
- mPLUG-owl-7B (Flamingo)
- GPT4-v
- Gemini Pro
- Pali3B
## Models used for the task (chart-trained VLLMs)
- ChartLLama
- ChartAssistant
- ChartInstruct
- ChartAst
## Chart specialist models
- ChartBERT
- Pix2Struct
- Matcha
- Unichart


## Transformation of graphs/charts (chart data extraction)
- OGR (Optical Graph Recognotion) / OCR (Optical character recognition) eg. ChartOCR model
- DePLot
- Matcha
- Pix2Struct
- StructChart

## Metrics (each with precision, recall and F1)
- BERTscore
- ROUGE-L
- ROUGE-1

## Links

dataset:
https://huggingface.co/datasets/katebor/SciVQA

papers:

dataset + methodology
SciGraphQA (part of sciVQA dataset)
https://arxiv.org/pdf/2308.03349

DePlot
https://arxiv.org/abs/2212.10505

Pix2Struct
https://arxiv.org/pdf/2210.03347v2

StructChart
https://arxiv.org/pdf/2309.11268v5

ChartQA
https://aclanthology.org/2022.findings-acl.177.pdf

Leadboard (Chart Question Answering on ChartQA):
https://paperswithcode.com/sota/chart-question-answering-on-chartqa

## First trials: zero-shot with Qwen2.5-VL in development set
with/without OCR (EasyOCR, Tesseract, Sunia)

## Second trials: fine-tuned model versions of Qwen2.5-VL with instructed data generated for the zero-shot

❌ Version 1: [config](LoRa_versions/Version_1/adapter_config.json)
change all the attention and MLP layers
we saved a checkpoint, we didn't finish the training

✅ Version 2: [config](LoRa_versions/Version_2/adapter_config.json)
as the tutorial: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl#2-load-dataset-
`"target_modules"=["q_proj", "v_proj"]`
change only two attention layers (query and value)


✅ Version 3: [config](LoRa_versions/Version_3/adapter_config.json)
`"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]`
change only attention layers (query, key, value, output) of the text decoder

✅ Version 4: [config](LoRa_versions/Version_4/adapter_config.json)
`"target_modules": ["up_proj", "gate_proj", "down_proj"]`
change all the MLP layers of the text decoder

✅ Version 5: [config](LoRa_versions/Version_5/adapter_config.json)
`"target_modules": ["layers.26.mlp.up_proj", "layers.27.mlp.down_proj"]`
change only final MLP layers of the text decoder

✅ Version 6:
`"target_modules": ["up_proj", "gate_proj", "down_proj", "q_proj", "v_proj"]`
change all MLP layers and query and value attention layer

🚧 Version 7:
`"target_modules": ["visual.blocks.X.attn.qkv", "visual.blocks.X.attn.proj"]`
change only attention layers of the visual encoder (visual encoder use a single qkv linear layer instead of separate   `q_proj`, `k_proj`, `v_proj`)

🚧 Version 8:
Try to target merge modules o visual encoder network layers ??

🚧 Version 9:
Try different adapters?

🚧 Version 10:
Same set up different models (see Leaderboard https://huggingface.co/spaces/opencompass/open_vlm_leaderboard )

🚧 Version 11:
same set up different data (ChartQA)

## Error analysis
[error_analysis/error_analysis_zeroshot_no-ocr-v4.xlxs](error_analysis/error_analysis_zeroshot_no-ocr-v4.xlsx): error analysis of the best zero-shot generated data

## Dataset:
- Unichart
- InstructChart
- ChartGemma
- ReachQA
- WebCharts
- ChartFC
- PlotQA

## Benchmark
- ChartQA
- Chart-to-Text
- Chart-to-Table
- OpenCQA
- ChartX