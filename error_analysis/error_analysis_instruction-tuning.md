## Possible sources of errors in instruction-tuning

# Data:
- **Not enough data**: use train set instead of dev set
- **Quality of data** is not good, not well-annotated and enough diverse: use ChartQA or other datasets instead of ours
- **Data format** does not match data format they used for training the model: check how data are formatting for training the model. Try formatting data closely how Qwen2.5-VL was trained (prompt structure and image handling)

#  Target modules:
- We are not targetting the **right modules**: language decoder is not a problem. Try to target vision decoder layers (networks or attention layers) or projector or cross-attention layers
- We are freezing too much of the model

# LoRa Adapters:
- **Lora Adapters** is not adequate: maybe because they already used it in modifying the original model in the latest version of Qwen2.5-V. Try with other adapters
- **ranking** and **alpha** are low: not our case

# The model is memorizing irrelevant patterns:
- If the training loss is decreasing but validation loss or accuracy is stagnant or worse. Try add some **distortions in the data** eg chart distortions (contrastive learning) or more varied questions (but I don't think this is the issue)

General impression I have: the things we are trying to modify on the model are too little and with not enough data and diverse data: so our fine-tuning is ineffective (Like throwing a bucket of water into a burning forest). The patterns that the model is learning are already there
