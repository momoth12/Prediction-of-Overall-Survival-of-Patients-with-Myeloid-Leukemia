# Prediction of Overall Survival of Patients with Myeloid Leukemia by QRT  

## Introduction  
This competitive challenge, organized in partnership with the Gustave Roussy Institute, focuses on predicting the risk of death for patients diagnosed with a subtype of adult myeloid leukemia. The goal is to develop predictive models to estimate patients' overall survival, a crucial piece of information for tailoring therapeutic strategies.  

## Background  
Predictive models in healthcare have revolutionized patient care, particularly in oncology. This challenge provides a unique opportunity to work with real-world data from 24 clinical centers and contribute to a practical application of data science in the medical field.  

## Objective  
Predict the overall survival (OS) of patients diagnosed with blood cancer. Participants must submit a CSV file containing:  
- **ID**: Unique patient identifier.  
- **risk_score**: Predicted risk of death.  

The performance metric used for evaluation is the **IPCW-C-index**, which measures the concordance of predictions with actual survival times.  

## Data Description  
### Clinical Data  
- **ID**: Unique identifier.  
- **CENTER**: Clinical center.  
- **BM_BLAST**: Percentage of blasts in bone marrow.  
- **WBC**: White blood cell count (Giga/L).  
- **ANC**: Absolute neutrophil count (Giga/L).  
- **MONOCYTES**: Monocyte count (Giga/L).  
- **HB**: Hemoglobin level (g/dL).  
- **PLT**: Platelet count (Giga/L).  
- **CYTOGENETICS**: Description of chromosomal abnormalities.  

### Molecular Data  
- **ID**: Unique identifier.  
- **CHR, START, END**: Chromosomal position of the mutation.  
- **REF, ALT**: Reference and alternate nucleotides.  
- **GENE**: Affected gene.  
- **PROTEIN_CHANGE**: Impact on the protein.  
- **EFFECT**: Classification of the impact.  
- **VAF**: Variant allele fraction.  

## Benchmarks  
Two models are provided:  
1. **LightGBM**: Using only clinical data.  
2. **Cox Model**: Incorporating both clinical and genetic data.  

The benchmark score is based on the Cox model.  

## Available Files  
- **x_train.zip**: Training explanatory variables.  
- **y_train.csv**: Training target variables.  
- **x_test.zip**: Testing explanatory variables.  
- **Random submission example**: Example CSV file.  

## Getting Started  

### Prerequisites  
To run the code and reproduce the results, you will need:  
- Python 3.8 or higher.  
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `lifelines`, and `matplotlib`.  

Install the dependencies using:  
```bash  
pip install -r requirements.txt  
```  

### Repository Structure  
- **data/**: Contains the training and testing datasets.  
- **notebooks/**: Jupyter notebooks for exploratory data analysis and model development.  
- **src/**: Source code for data preprocessing, feature engineering, and model training.  
- **models/**: Saved models and benchmark results.  
- **submission/**: Example submission files.  

### Running the Code  
1. **Data Preparation**:  
    Unzip the `x_train.zip` and `x_test.zip` files into the `data/` directory.  

2. **Training the Model**:  
    Run the training script:  
    ```bash  
    python src/train_model.py  
    ```  

3. **Generating Predictions**:  
    Use the trained model to generate predictions:  
    ```bash  
    python src/predict.py  
    ```  

4. **Submission**:  
    Ensure your predictions are in the required format and submit the CSV file.  

### Evaluation  
The model's performance is evaluated using the **IPCW-C-index** metric. Refer to the `notebooks/evaluation.ipynb` for details on how to compute this metric.  

### Contribution  
Feel free to contribute by submitting pull requests or reporting issues.  

### License  
This project is licensed under the MIT License. See the `LICENSE` file for details.  