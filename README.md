# Event Classifier using Hugging Face Zero-Shot Classification (with the `facebook/bart-large-mnli` model)
This repository has a script that classifies events based on if they are available to Canadians (virtual or in-Canada) and open to the general public. The script uses an NLP method called zero-shot classification.
### Zero-Shot Classification:
* the task of predicting a class that wasn't seen by the model during training. 
* leverages a pre-trained language model, can be thought of as an instance of transfer learning which generally refers to using a model trained for one task in a different application than what it was originally trained for. 
* particularly useful for situations where the amount of labeled data is small (like our case!)

## Setup and Usage
### 1. Prerequisites
- Python 3.6+
- `pip` for python packages

### 2. Installation
1. Clone the repository.
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes:
    - `transformers`: For Hugging Face's model pipelines
    - `torch`: PyTorch backend for model execution
    - `pandas`: For processing the CSV file

### 3. Running the Script
1. Place your CSV file (e.g., `events.csv`) in the project directory.
2. Run the script:
    ```bash
    python main.py
    ```

### 4. Model Download
- The first time you run the script, it will automatically download the `facebook/bart-large-mnli` model from Hugging Face's model hub.
- **Note**: This download can take a few minutes depending on your internet connection, as the model size is approximately 1.6 GB.
* The model files are stored in a cache directory on your system.
* These files are not stored in the project directory and, therefore, are not included when you push your project to a Git repository.

### Using the Script
- The script reads the CSV file, concatenates relevant columns, and uses the zero-shot classification pipeline to determine if each event is "available to Canadians" or "open to the general public".
- The results are printed to the console, showing the classification labels and scores for each event.