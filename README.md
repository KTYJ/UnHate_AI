# UnHate_AI
Take control of your online world. Our friendly intelligent app helps you build a more positive space by proactively identifying and flagging hateful comments and cyberbullying across multiple languages. It's time to foster conversations, not conflicts.

## 📁 Project Structure

The repository is organized as follows:
- **Primary Files**
    - **`main_st_noroberta.py`**: Main Streamlit file of the application 
    - **`BMCS2074_UnhateAI.ipynb`**: A Jupyter Notebook containing the primary code for data preprocessing, model training, and evaluation.
    - **`requirements.txt`**: A list of all the Python libraries and dependencies required to run the project.

- **Prediction Model Files**
    - **`xlm-roberta-large-model/`**: (not available) Directory containing the fine-tuned XLM-RoBERTa model. (not available)
    - **`xlm-roberta-results/`**: (not available) Directory to store results and outputs from the XLM-RoBERTa model.
    - **`balanced_dataset.csv`**: The primary dataset used for training and evaluation, which has been balanced to handle class imbalances.
    - **`cnn_text_classifier.h5`**: The trained Convolutional Neural Network model saved in HDF5 format.
    - **`cnn_tokenizer.json`**: The tokenizer for the CNN model, saved in JSON format.
    - **`label_encoder.pkl`**: A serialized LabelEncoder object used to transform categorical labels into a numerical format.
    - **`nb_finetuned_model.joblib`**: The fine-tuned Naive Bayes model saved using joblib.
    - **`svm_finetuned_model.joblib`**: The fine-tuned Support Vector Machine model saved using joblib.
    - **`tfidf.dill`**: The TF-IDF vectorizer saved using the dill library

- **Other**:
    - `dataset-cover.jpg`: A cover image for the dataset.
    - **`.idea/`**: Project configuration files for JetBrains IDEs.

## 🤖 Models Used

This project explores several different models for text classification:

*   **Convolutional Neural Network (CNN)**: A deep learning model that can be effective for text classification tasks.
*   **Support Vector Machine (SVM)**: A robust machine learning algorithm that is well-suited for high-dimensional data like text.
*   **Naive Bayes**: A probabilistic classifier that is often used as a baseline for text classification.
*   **XLM-RoBERTa**: (not available) A powerful multilingual transformer model that has been fine-tuned for this specific task. (not available in GitHub Version due to large size)

## 🤔 Getting Started

### Prerequisites

To run this project, you will need to have Python and the libraries listed in the `requirements.txt` file installed.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The workflow can be found in the `BMCS2074_UnhateAI.ipynb` Jupyter Notebook. You can open the notebook to see the data preprocessing, model training, and evaluation steps.

#### OR
**Run this in the root directory of the folder:**
```bash
streamlit run main_st.py
```

#### OR
**Open this website in your browser:**
```bash
https://unhateai-ktyj.streamlit.app/
```

