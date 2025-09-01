# 🛡️ AI/ML Compliance Audit Tool


This repository contains the first version of an **AI/ML Compliance Audit Tool** designed to evaluate whether machine learning pipelines deployed in **AWS** comply with **GDPR** requirements.  

![Architecture](./aws_ml_compliance_architecture.jpg)

The tool provides both:
- **Automated auditing logic** (`audit.py`)  
- **Interactive dashboard interface** (`dashboard.py`)  


## 🎯 Motivation
The growing adoption of AI systems raises **compliance and ethical challenges**. Regulations like the **GDPR** require that organizations ensure:  
- Data is stored and transferred responsibly  
- AI systems are **traceable, fair, and explainable**  
- Users’ rights and organizational accountability are respected  

This tool demonstrates how **compliance checks** can be integrated into the ML pipeline lifecycle.  



## 📊 Dashboard Preview
*Placeholder for demo GIF of dashboard interface:*  
![Dashboard Demo](demo.gif)  


## 🔎 Compliance Checks Implemented

### 1. **Data Residency**  
- GDPR restricts cross-border data transfers (Art. 44–46). Data must stay within approved regions (e.g., EU).  
- The tool verifies S3 bucket locations and ensures data is encrypted.  

### 2. **Traceability & Logging**  
- GDPR (Art. 5(2), 30) requires **accountability and auditability**.  
- The tool logs which models are linked to which datasets, buckets, and training jobs. This allows reconstruction of the pipeline for auditing.  

### 3. **Fairness & Bias Metrics**  
- GDPR (Art. 5(1)(a), Recital 71) requires fairness and non-discrimination in automated decisions.  
- Checked using AWS Clarify metrics:  
  - **Accuracy Difference (AD):** Difference in accuracy between groups  
  - **Difference in Positive Proportions in Predicted Labels (DPPL):** Measures group-wise disparities  
  - **Difference in Acceptance Rates (DAR):** Compares acceptance ratios  
  - **Treatment Equality (TE):** Ratio of false negatives to false positives across groups  

### 4. **Explainability**  
- GDPR (Art. 22, Recital 71) gives users the right to meaningful explanations of automated decisions.  
- The tool integrates **SHAP (SHapley Additive exPlanations)** to explain model predictions at both global and local levels.  


## ⚙️ Usage

1. Clone & Install
```bash
git clone https://github.com/yourusername/compliance-audit-tool.git
cd compliance-audit-tool
pip install -r requirements.txt 
```
2. Run Compliance Audit
```
python audit.py --bucket <bucket_name> --model <model_name>
```
3. Launch Dashboard
```
streamlit run dashboard.py
```



## 📂 Repository Structure
```
compliance-audit-tool/
│
├── audit.py             # Core compliance checks
├── dashboard.py         # Interactive dashboard
├── README.md            # Project 
```


## 🚀 Future Updates

Planned improvements and extensions to the audit tool include:
-	Expanded ML Use Cases – Support for a broader range of machine learning pipelines beyond toy examples and classification tasks.
-	Extended GDPR Coverage – Deeper checks across multiple GDPR articles and compliance dimensions (consent, policies, access logs, and transfers).
-	Access Control & Policy Auditing – Verification of user permissions, IAM roles, and policy enforcement.
-	Data Collection & Consent Validation – Ensuring proper consent mechanisms and lawful basis for data use.
-	Risk Analysis – Identifying and quantifying compliance and ethical risks.
-	Support for NLP Models – Extending coverage beyond structured data classification to natural language processing use cases.
-	LLM-based Assistant – Adding a layer for generating summaries, remediation suggestions, and compliance insights using large language models.
-	Testing & Types – Comprehensive unit tests, type checking, and improved developer experience.
-	Improve dashboard with real-time monitoring and alerts


## 📖 References
-	[GDPR Full Text](https://gdpr-info.eu/)
-	[AWS SageMaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-configure-processing-jobs.html)
-	Lundberg & Lee, A Unified Approach to Interpreting Model Predictions, [NeurIPS 2017 (SHAP)](http://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

---
💡 Note: This is an early-stage prototype. More compliance dimensions will be added in future iterations. Contributions and feedback are welcome!
