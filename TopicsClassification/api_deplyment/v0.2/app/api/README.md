topic_classification.py
# Topic Classification with BERT

This Python script, `topic_classification.py`, is designed to perform topic classification on a given text input using a pre-trained BERT model. The script loads the necessary model, tokenizer, and encoding files, and defines a custom model class to perform multi-label classification. It also includes a function to analyze the text and return the predicted topics along with their scores.

### Dependencies

The script requires the following libraries:

- numpy
- transformers
- torch

### Usage

1. Set the required parameters in the `parameter.json` file, including the paths to the model, encoding, BERT model, tokenizer, GPU device, and threshold for topic classification.

2. Load the encoding file using the specified path in the `parameter.json` file.

3. Define the custom BERT model class `mybert` which inherits from `nn.Module`. This class includes a pre-trained BERT model and additional layers for multi-label classification.

4. Load the pre-trained BERT model and tokenizer using the paths specified in the `parameter.json` file.

5. Define the sigmoid function and set the model to evaluation mode.

6. Define the `analyze_text_topics_inhouse(text)` function which takes a text input, tokenizes it, and feeds it to the custom BERT model. The function returns the predicted topics and their scores in JSON format.

7. (Optional) Uncomment the demo data and function call at the end of the script to test the topic classification on a sample text.

### Example

```python
text = "Make the devices able to be setup without requiring an internet connection"
result = analyze_text_topics_inhouse(text)
print(result)
```

This will return a JSON object containing the input text, predicted topics, and their scores.

### Note

Please ensure that the paths specified in the `parameter.json` file are correct and that the required files are available before running the script.