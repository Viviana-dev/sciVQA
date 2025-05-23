❌ Version 1: [config](LoRa_versions/Version_1/adapter_config.json)\
change all the attention and MLP layers\
we saved a checkpoint, we didn't finish the training

✅ Version 2: [config](LoRa_versions/Version_2/adapter_config.json)\
as the tutorial: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl#2-load-dataset-\
`"target_modules"=["q_proj", "v_proj"]`
change only two attention layers (query and value)

✅ Version 3: [config](LoRa_versions/Version_3/adapter_config.json)\
`"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]`\
change only attention layers (query, key, value, output) of the text decoder

✅ Version 4: [config](LoRa_versions/Version_4/adapter_config.json)\
`"target_modules": ["up_proj", "gate_proj", "down_proj"]`\
change all the MLP layers of the text decoder

✅ Version 5: [config](LoRa_versions/Version_5/adapter_config.json)\
`"target_modules": ["layers.26.mlp.up_proj", "layers.27.mlp.down_proj"]`\
change only final MLP layers of the text decoder

✅ Version 6 & 7: [config 6](LoRa_versions/Version_6/adapter_config.json) [config 7](LoRa_versions/Version_7/adapter_config.json)\
`"target_modules": ["up_proj", "gate_proj", "down_proj", "q_proj", "v_proj"]`\
change all MLP layers and query and value attention layer

✅ Version 8: [config](LoRa_versions/Version_8/adapter_config.json)\
`"target_modules": ["v_proj", "up_proj", "gate_proj", "down_proj", "q_proj", "k_proj"]`\
Add 10% White padding around the Image

✅ Version 9: [config](LoRa_versions/Version_9/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj", "down_proj", "visual.blocks.X.attn.proj", "visual.blocks.X.attn.qkv"]`\
Update Prompt

✅ Version 10 & 11: [config 10](LoRa_versions/Version_10/adapter_config.json) [config 11](LoRa_versions/Version_11/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "visual.blocks.X.attn.proj", "visual.blocks.X.attn.qkv"]`\
Add ChartQA dataset to train

✅ Version 12: [config](LoRa_versions/Version_12/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
Skip ChartQA again

✅ Version 13: [config](LoRa_versions/Version_13/adapter_config.json) <- Gets very bad results\
`"target_modules": ["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
add ChartQA again

✅ Version 14: [config](LoRa_versions/Version_14/adapter_config.json)\
`"target_modules": ['q_proj', 'v_proj', 'o_proj', 'k_proj', 'up_proj', 'gate_proj', 'down_proj', 'proj', 'qkv']`\
remove ChartQA add OCR with new specialtoken `<box>` and `<\box>`\
Save also the Processor now

✅ Version 15: [config](LoRa_versions/Version_15/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`

✅ Version 16: [config](LoRa_versions/Version_16/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`

✅ Version 16: [config](LoRa_versions/Version_16/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`

✅ Version 17: [config](LoRa_versions/Version_17/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
Finetune Gemma3-8B-it --> Fail --> Worst results

✅ Version 18: [config](LoRa_versions/Version_18/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Try with rerurnnin and providing prvious answers

✅ Version 19: [config](LoRa_versions/Version_19/adapter_config.json)\
`"target_modules": ["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "proj", "qkv"]`\
Add CoT - remove OCR - remove retraining

✅ Version 20: [config](LoRa_versions/Version_20/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Add Acceleration - better distributed training

✅ Version 21: [config](LoRa_versions/Version_21/adapter_config.json) <-- Best model - Latest for leaderboard \
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Update CoT

QwenChart 7B

✅ Version 22: [config](LoRa_versions/Version_22/adapter_config.json)\
`"target_modules": ["mlp.0", "mlp.2", "qkv", "attn.proj", "gate_proj", "up_proj", "q_proj", "v_proj", "k_proj", "down_proj","o_proj"]`\
72B model,targe_modules = "all-linear"

QwenChart 72B

✅ Version 23: [config](LoRa_versions/Version_23/adapter_config.json)\
`"target_modules": "^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"`\
Update Instruction to be more general

QwenChart2

✅ Version 24: \
Test the Version 23 Model with ChartQA dataset

QweChart2 on ChartQA

✅ Version 25: \
Zero-shot with Gemma3-12b-it\

✅ Version 26: \
Zero-shot with Qwen2.5-VL-7B-Instruct



## Error analysis
[error_analysis/error_analysis_zeroshot_no-ocr-v4.xlxs](error_analysis/error_analysis_zeroshot_no-ocr-v4.xlsx): error analysis of the best zero-shot generated data


## Notes:
Target Modules: ^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*\
trainable params: 1,248,629,760 || all params: 9,537,950,720 || trainable%: 13.0912

Target Modules: ['q_proj', 'v_proj', 'o_proj', 'k_proj', 'up_proj', 'gate_proj', 'down_proj', 'proj', 'qkv']\
trainable params: 1,293,392,384 || all params: 9,582,713,344 || trainable%: 13.4971

Target Modules: ['q_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']\
trainable params: 1,257,321,472 || all params: 9,546,642,432 || trainable%: 13.1703
