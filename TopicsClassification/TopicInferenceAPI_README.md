# Topic Inference API

This Python script, `TopicInferenceAPI.py`, is designed to analyze a given text and infer the topics it contains using a pre-trained XLM-RoBERTa model. The script loads the model, tokenizes the input text, and processes it to generate topic predictions along with their associated scores.

### Dependencies

- numpy
- torch
- transformers

### Usage

1. Replace the paths in the code with your actual paths for the `parameter.json`, `model_path`, and `encoding_path`.
2. Run the script with the desired input text.

### Key Components

- `MyBert`: A custom PyTorch module that extends the `nn.Module` class. It contains a pre-trained XLM-RoBERTa model and additional layers for multi-label classification.
- `analyze_text_topics_inhouse(text)`: A function that takes a text input, tokenizes it, and processes it through the model to generate topic predictions and their associated scores.

### Example

```python
text = "I dont care, it's a bad product. I don't want to use it anymore"
analyze_text_topics_inhouse(text)
```

This will output a JSON object containing the input text, the predicted topics, and their associated scores.

### Notes

- The `parameter.json` file should contain the following keys: `model_path`, `encoding_path`, `gpu_device`, and `threshold`.
- The `model_path` should point to the pre-trained XLM-RoBERTa model file.
- The `encoding_path` should point to the file containing the topic encodings.
- The `gpu_device` should be set to the desired GPU device index for running the script.
- The `threshold` should be set to the desired threshold value for topic prediction scores.