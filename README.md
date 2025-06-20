# QwenChart

This repository contains the code and data for the **QwenChart** project, which participated in the [SciVQA 2025 Shared Task](https://sdproc.org/2025/scivqa.html). The model is based on fine-tuning [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) using instruction-tuning for scientific visual question answering.

## Ressources
- **Dataset:** [SciVQA](https://huggingface.co/datasets/katebor/SciVQA)
- **Base Model:** [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Shared Task:** [SciVQA 2025](https://sdproc.org/2025/scivqa.html)

## Model Overview
QwenChart is fine-tuned to understand and answer questions about scientific charts and graphs. It leverages:
- **Instruction tuning** using the [SciVQA dataset](https://huggingface.co/datasets/katebor/SciVQA)
- **LoRA adapters** for parameter-efficient fine-tuning
- **Chain-of-Thought (CoT)** prompting
- **Error analysis and iterative improvements** across multiple configurations


## Model Development
QwenChart has undergone extensive experimentation through 26 LoRA-based instructen-tuning versions. Key modifications included adjustments to target modules, inclusion/exclusion of OCR and auxiliary datasets, prompt engineering, and distributed training enhancements.

### Best performing model
The best performing model is **Version 21** with the following configuration:\
**Target Modules**: `^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*`\
[**Config**](LoRa_versions/Version_21/adapter_config.json) | [**Scores**](Scores_versions/Version_21/scores.txt)

## Version History
A full history of all fine-tuning experiments with configuration links and performance metrics is available below:
<details>
<summary>Click to expand full version history</summary>

❌ Version 1: [config](LoRa_versions/Version_1/adapter_config.json)\
change all the attention and MLP layers\
we saved a checkpoint, we didn't finish the training

✅ Version 2: [config](LoRa_versions/Version_2/adapter_config.json)\
as the tutorial: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl#2-load-dataset-\
`"target_modules"=["q_proj", "v_proj"]`
change only two attention layers (query and value)\
[Scores](Scores_versions/Version_2/scores.txt)

✅ Version 3: [config](LoRa_versions/Version_3/adapter_config.json)\
`"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]`\
change only attention layers (query, key, value, output) of the text decoder\
[Scores](Scores_versions/Version_3/scores.txt)

✅ Version 4: [config](LoRa_versions/Version_4/adapter_config.json)\
`"target_modules": ["up_proj", "gate_proj", "down_proj"]`\
change all the MLP layers of the text decoder\
[Scores](Scores_versions/Version_4/scores.txt)

✅ Version 5: [config](LoRa_versions/Version_5/adapter_config.json)\
`"target_modules": ["layers.26.mlp.up_proj", "layers.27.mlp.down_proj"]`\
change only final MLP layers of the text decoder\
[Scores](Scores_versions/Version_5/scores.txt)

✅ Version 6 & 7: [config 6](LoRa_versions/Version_6/adapter_config.json) [config 7](LoRa_versions/Version_7/adapter_config.json)\
`"target_modules": ["up_proj", "gate_proj", "down_proj", "q_proj", "v_proj"]`\
change all MLP layers and query and value attention layer\
[Scores 6](Scores_versions/Version_6/scores.txt) [Scores 7](Scores_versions/Version_7/scores.txt)

✅ Version 8: [config](LoRa_versions/Version_8/adapter_config.json)\
`"target_modules": ["v_proj", "up_proj", "gate_proj", "down_proj", "q_proj", "k_proj"]`\
Add 10% White padding around the Image\
[Scores](Scores_versions/Version_8/scores.txt)

✅ Version 9: [config](LoRa_versions/Version_9/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj", "down_proj", "visual.blocks.X.attn.proj", "visual.blocks.X.attn.qkv"]`\
Update Prompt\
[Scores](Scores_versions/Version_9/scores.txt)

✅ Version 10 & 11: [config 10](LoRa_versions/Version_10/adapter_config.json) [config 11](LoRa_versions/Version_11/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "visual.blocks.X.attn.proj", "visual.blocks.X.attn.qkv"]`\
Add ChartQA dataset to train\
[Scores 10](Scores_versions/Version_10/scores.txt) [Scores 11](Scores_versions/Version_11/scores.txt)

✅ Version 12: [config](LoRa_versions/Version_12/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
Skip ChartQA\
[Scores](Scores_versions/Version_12/scores.txt)

✅ Version 13: [config](LoRa_versions/Version_13/adapter_config.json) <- Gets very bad results\
`"target_modules": ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
add ChartQA again\
[Scores](Scores_versions/Version_13/scores.txt)

✅ Version 14: [config](LoRa_versions/Version_14/adapter_config.json)\
`"target_modules": ['q_proj', 'v_proj', 'o_proj', 'k_proj', 'up_proj', 'gate_proj', 'down_proj', 'proj', 'qkv']`\
remove ChartQA add OCR with new specialtoken `<box>` and `<\box>`\
Save also the Processor now\
[Scores](Scores_versions/Version_14/scores.txt)

✅ Version 15: [config](LoRa_versions/Version_15/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
[Scores](Scores_versions/Version_15/scores.txt)

✅ Version 16: [config](LoRa_versions/Version_16/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
[Scores](Scores_versions/Version_16/scores.txt)

✅ Version 17: [config](LoRa_versions/Version_17/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
Finetune Gemma3-8B-it --> Fail --> Worst results\
[Scores](Scores_versions/Version_17/scores.txt)

✅ Version 18: [config](LoRa_versions/Version_18/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Try with retraining and providing prvious answers -> Fail\
[Scores](Scores_versions/Version_18/scores.txt)

✅ Version 19: [config](LoRa_versions/Version_19/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
Add CoT - remove OCR - remove retraining\
[Scores](Scores_versions/Version_19/scores.txt)

✅ Version 20: [config](LoRa_versions/Version_20/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Add Acceleration - better distributed training\
[Scores](Scores_versions/Version_20/scores.txt)

✅ Version 21: [config](LoRa_versions/Version_21/adapter_config.json) <-- Best model - Latest for leaderboard \
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Update CoT\
[Scores](Scores_versions/Version_21/scores.txt)

✅ Version 22: [config](LoRa_versions/Version_22/adapter_config.json)\
`"target_modules": ["mlp.0", "mlp.2", "qkv", "attn.proj", "gate_proj", "up_proj", "q_proj", "v_proj", "k_proj", "down_proj","o_proj"]`\
72B model, targe_modules = "all-linear"\
[Scores](Scores_versions/Version_22/scores.txt)

✅ Version 23: [config](LoRa_versions/Version_23/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Update Instruction to be more general\
[Scores](Scores_versions/Version_23/scores.txt)

✅ Version 24: \
Test the Version 23 Model with ChartQA dataset\
[Scores](Scores_versions/Version_24/scores.txt)

✅ Version 25: \
Zero-shot with Gemma3-12b-it\
[Scores](Scores_versions/Version_25/scores.txt)

✅ Version 26: \
Zero-shot with Qwen2.5-VL-7B-Instruct\
[Scores](Scores_versions/Version_26/scores.txt)
</details><br>


Each version documents:
- `target_modules` used in LoRA
- Dataset or method changes (e.g., ChartQA, OCR, CoT)
- Score file links


## Error analysis
A systematic error analysis was conducted on early model versions to guide improvements.\
[Error Analysis (Excel) Version 4](error_analysis/error_analysis_zeroshot_no-ocr-v4.xlsx)

## Target Modules Notes:
Target Modules: `^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*`\
trainable params: 1,248,629,760\
all params: 9,537,950,720\
trainable: 13.0912%

Target Modules: `['q_proj', 'v_proj', 'o_proj', 'k_proj', 'up_proj', 'gate_proj', 'down_proj', 'proj', 'qkv']`\
trainable params: 1,293,392,384\
all params: 9,582,713,344\
trainable: 13.4971%

Target Modules: `['q_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']`\
trainable params: 1,257,321,472\
all params: 9,546,642,432\
trainable: 13.1703%

## Citation
If you use this work, please cite our shared task paper (citation will be added after camera-ready version).
