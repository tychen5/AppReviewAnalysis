# Topic Classification with XLM-RoBERTa

This Python script is designed to classify text into topics using a pre-trained XLM-RoBERTa model. The script loads a custom model, tokenizes input text, and predicts the topics based on a given threshold. The output is a JSON object containing the input text, predicted topics, and their associated scores.

### Dependencies

- numpy
- transformers
- torch
- os
- pickle
- json

### Usage

To use this script, you need to provide the following:

1. A JSON file containing the parameters:
   - `model_path`: Path to the pre-trained model
   - `encoding_path`: Path to the encoding file
   - `gpu_device`: GPU device to use
   - `threshold`: Threshold for topic classification

2. A text string to be analyzed for topic classification.

### Custom Model

The script defines a custom model class `MyBERT` that inherits from `nn.Module`. This class uses the pre-trained XLM-RoBERTa model and adds additional layers for multi-label classification.

### Functions

- `analyze_text_topics_inhouse(text)`: This function takes a text string as input, tokenizes it, and predicts the topics using the custom model. The output is a JSON object containing the input text, predicted topics, and their associated scores.

### Example

```python
text = "I dont care, it's a bad product. I don't want to use it anymore"
result = analyze_text_topics_inhouse(text)
```

The `result` variable will contain a JSON object with the input text, predicted topics, and their associated scores.