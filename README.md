# Titanic Survival Classification (CRISP-DM)

End-to-end **machine learning** project on the Kaggle **Titanic** dataset following the **CRISP-DM** framework:
data preparation, exploration, modeling with multiple classifiers, evaluation, model selection, and an ethics
discussion.

## Run in Google Colab

- **Data Exploration**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1spZrRnXrN3rpwVUjeTxCeyptX49Uweqi?usp=sharing)

- **Data Preparation**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LKKUAoL6DZl3TCmTRGXkL18ryKiFDYHF?usp=sharing)

- **Data Modeling**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rAtA8rNuVeQ2PM4f7Qx0lOmbHFszN1Vp?usp=sharing)

---

## 🎯 Objective
Predict whether a passenger survived the Titanic disaster using tabular features (demographics, ticket class, fare, cabin/embarkation info).

## 🧭 CRISP-DM Phases
1. **Business Understanding** – Predict survival; compare models and justify the best one.
2. **Data Understanding** – Inspect missingness (e.g., `Age`, `Cabin`), distributions, and survival correlations.
3. **Data Preparation** – Impute, encode categoricals (`Sex`, `Embarked`, `Pclass`), feature engineer (e.g., `Title`, `FamilySize`, `IsAlone`), scale where needed.
4. **Modeling** – Train and tune multiple classifiers:
   - Logistic Regression, Linear Discriminant Analysis (LDA)
   - Support Vector Machine (SVM)
   - Random Forest, Gradient Boosting
   - k-Nearest Neighbors (KNN)
5. **Evaluation** – Use stratified train/test or CV; metrics: Accuracy, F1, ROC-AUC, confusion matrices.
6. **Deployment (lightweight)** – Produce `submission.csv` for Kaggle from the best model.

---

## 📁 Repository Structure
notebooks/
├── 01_DataPreparation.ipynb # Cleaning, imputing, encoding, feature engineering
├── 02_DataExploration.ipynb # EDA and insights about survival
├── 03_DataModeling.ipynb # Multiple classifiers + cross-validation
└── 04_Submission.ipynb # Train best model on full train, predict test.csv
reports/
└── summary_paper.pdf # Final write-up including CRISP-DM discussion and ethics
data/
└── README.md # Instructions to download train.csv/test.csv from Kaggle
requirements.txt # Python dependencies


---

## 🛠️ Technologies Used

- **Python 3.x**
- **Google Colab** (or Jupyter Notebook)
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## ▶️ How to Run
Open in Google Colab
Click the badge above and run cells in order.  


