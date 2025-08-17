import streamlit as st
import pandas as pd
from audit import AuditClass

# --- Test input ---
input_dict = {
    'model_name': "xgboost-loan-prediction-clarify-29-7-2119",
    'model_data': 's3://herixai-ml/demo-sagemaker-loan-prediction/xgboost_model/sagemaker-xgboost-2025-07-29-19-12-41-589/output/model.tar.gz',
    'train_data_path': 's3://herixai-ml/demo-sagemaker-loan-prediction/data/train.csv',
    'data_type': "text/csv",
    'train_headers': ["Loan_Status", "Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"],
    'label': "Loan_Status",
    'fairness_setting': {'facet_name': "Gender"}
}

audit_bucket = "Audit_bucket_name"
audit_setting = {
    'auditor': "arn:aws:iam::12334568982:role/service-role/AmazonSageMaker-ExecutionRole",
    'instance_type': "ml.t3.medium",
    'instance_count': 1,
    'accept_type': "text/csv",
    'audit_path': f"s3://{audit_bucket}/audit_directory"
}

# --- Show test input in the dashboard ---
st.sidebar.header("Test Input")
st.sidebar.json(input_dict)
st.sidebar.header("Audit Settings")
st.sidebar.json(audit_setting)

# --- Run audit and show results as before ---
complianceEngine = AuditClass(input_dict, audit_setting)
audit_result_dict = complianceEngine.get_results()

st.title("🛡️ ML Pipeline Audit Report")
st.markdown(
    f""" 
    **Model Name:** `{audit_result_dict['model_name']}`   
    **Audit Date:** {audit_result_dict['audit_date']}  
    **Audit by:** `{audit_setting['auditor']}`
    """
)

# Data Residency Section
residency_status = "Approved ✅" if audit_result_dict['residency_check']['status'] else "Not Approved ⚠️"
residency_overview = "\n".join(audit_result_dict['residency_check']['overview'])
residency_suggestions = "\n".join(audit_result_dict['residency_check']['suggestion'])

data_residency = pd.DataFrame([
    {
        "Bucket": b['bucket_name'],
        "Location": b['location']['value'] + (" ⚠️" if not b['location']['is_eu'] else ""),
        "Encryption": (b['encryption']['type'] if b['encryption']['is_encrypted'] else "None ⚠️")
    }
    for b in audit_result_dict['residency_check']['s3_buckets']
])


residency_articles="""
- GDPR requires personal data to stay within the EU (unless strict safeguards are in place, Art. 44–50) and to be encrypted for security (Art. 32). Together, these ensure that personal data is not only legally protected by jurisdiction but also technically safeguarded.
"""

with st.container(border=True):
    st.header("📦 **Data Residency**")
    st.warning(f"⚠️ {residency_status}" if "Not Approved" in residency_status else f"✅ {residency_status}")
    st.dataframe(data_residency, hide_index=True)
    st.markdown(residency_overview)
    st.subheader("💡 Suggestions:")
    st.markdown(residency_suggestions)
    with st.expander("GDPR Art. 44–50 and Art. 32"):
        st.write(residency_articles)

# Traceability Section
traceability_status = "Approved ✅" if audit_result_dict['traceability_check']['status'] else "Not Approved ⚠️"
traceability_overview = "\n".join(audit_result_dict['traceability_check']['overview'])
traceability_suggestions = "\n".join(audit_result_dict['traceability_check']['suggestion'])

traceability_meta = audit_result_dict['traceability_check']['model_meta_data']
traceability_df = pd.DataFrame([
    {"Metadata": k, "Value": v['value'] if isinstance(v, dict) else v}
    for k, v in traceability_meta.items()
])

traceability_articles="""
- Traceability is essential under the GDPR because it enables organizations to demonstrate compliance with the accountability principle (Article 5(2)), ensuring that every step of data processing and model decision-making can be tracked and justified. It supports the obligation to maintain records of processing activities (Article 30) and to implement appropriate technical and organizational measures (Article 24) that guarantee lawful and transparent data use. By providing a clear audit trail of data lineage and model behavior, traceability also reinforces the GDPR’s emphasis on transparency and trust (Recital 74, 78), making AI systems explainable, auditable, and accountable.
"""

with st.container(border=True):
    st.header("🧾 **Traceability**")
    st.success(f"✅ {traceability_status}" if "Approved" in traceability_status else f"⚠️ {traceability_status}")
    st.dataframe(traceability_df, hide_index=True)
    st.markdown(traceability_overview)
    if traceability_suggestions:
        st.subheader("💡 Suggestions:")
        st.markdown(traceability_suggestions)
    with st.expander("GDPR Art. 5(2) and Art. 30"):
        st.write(traceability_articles)

# Explainability Section
explainability_status = "Approved ✅" if audit_result_dict['explainability_check']['status'] else "Not Approved ⚠️"
explainability_overview = "\n".join(audit_result_dict['explainability_check']['overview'])
explainability_suggestions = "\n".join(audit_result_dict['explainability_check']['suggestion'])

exp_info = audit_result_dict['explainability_check']['exp_info']
global_importance = exp_info['global_importance'] if exp_info else {}
local_importance = exp_info['local_importance'] if exp_info else {}


explainability_articles="""
- Under the GDPR, individuals have the right to understand how their personal data is processed and how automated decisions affecting them are made. Article 15(1)(h) grants the right of access to “meaningful information about the logic involved” in automated processing, while Article 22 and Recital 71 emphasize the need for transparency and safeguards in automated decision-making. This ensures accountability, builds trust, and allows data subjects to challenge or contest unfair or harmful outcomes.
"""

explainability_shap_help="""
- ** SHAP (SHapley Additive exPlanations)**: A game-theoretic method for explainability that attributes each feature’s contribution to a model’s prediction. It works by fairly distributing the “credit” (or blame) among features, helping users understand why a model made a specific decision.
"""

with st.container(border=True):
    st.header("📊 **Explainability**")
    st.success(f"✅ {explainability_status}" if "Approved" in explainability_status else f"⚠️ {explainability_status}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Global")
        st.markdown(" **Explainer Method:** SHAP (KernelExplainer)  ")
        st.markdown(" **Global Feature Attributions:** ")
        if global_importance:
            st.bar_chart(global_importance)
    with col2:
        st.subheader("Local")
        if local_importance:
            st.markdown(f"**Local Explanation (Example ID: {local_importance.get('example_id', 1)}):**")
            st.markdown(f"- **Predicted credibility Score:** `{local_importance.get('y_hat', '')}`")
            st.markdown(" **Local Feature Attributions:** ")
            st.bar_chart(local_importance.get('local_importance', {}), horizontal=True)
    st.markdown(explainability_overview)
    st.subheader("💡 Suggestions:")
    st.markdown(explainability_suggestions)
    with st.expander("GDPR Art. 15(1)(h), Art. 22, Recital 71"):
        st.write(explainability_articles)
    with st.expander("What are SHAP values?"):
        st.markdown(explainability_shap_help)

# Bias Section
bias_status = "Approved ✅" if audit_result_dict['bias_check']['status'] else "Not Approved ⚠️"
bias_overview = "\n".join(audit_result_dict['bias_check']['overview'])
bias_suggestions = "\n".join(audit_result_dict['bias_check']['suggestion'])
bias_info = audit_result_dict['bias_check']['bias_info']

bias_df = pd.DataFrame([
    {"Field": k, "Value": v['value'] if isinstance(v, dict) else v}
    for k, v in bias_info.items()
])


bias_articles="""
- GDPR enshrines the principle of fair and lawful processing (Article 5(1)(a)), requiring that individuals are not subject to unjust or discriminatory treatment through automated decision-making. Specifically, Article 22 gives data subjects the right not to be subject to solely automated decisions that have legal or significant effects, highlighting risks of biased or unfair outcomes. Moreover, Recitals 71 and 72 stress that safeguards must prevent discrimination based on sensitive attributes (such as race, gender, or religion), making bias detection and mitigation a key compliance requirement for AI and ML systems.
"""

bias_metrics_help="""
- Accuracy Difference (AD): Measures the difference in prediction accuracy between groups (e.g., male vs. female). A high value indicates unequal model performance across groups.
- Difference in Positive Proportions in Predicted Labels (DPPL): Compares how often different groups receive positive predictions. Large gaps suggest potential bias in outcomes.
- Difference in Acceptance Rates (DAR): Shows the disparity in the rate of positive outcomes (e.g., loan approvals) across groups. Used to assess fairness of decisions.
- Treatment Equality (TE): Evaluates fairness in terms of false positive and false negative rates between groups, highlighting unequal error distribution.
"""

with st.container(border=True):
    st.header("⚖️ **Bias and Fairness**")
    st.warning(f"⚠️ {bias_status}" if "Not Approved" in bias_status else f"✅ {bias_status}")
    st.dataframe(bias_df, hide_index=True)
    st.markdown(bias_overview)
    st.subheader("💡 Suggestions:")
    st.markdown(bias_suggestions)
    with st.expander("GDPR Art. 5(1)(a), Art. 22"):
        st.write(bias_articles)
    with st.expander("What are bias metrics?"):
        st.write(bias_metrics_help)

st.markdown(
    f"""
    > Audit data are stored in : `{audit_result_dict['audit_log']}`

    > _This report is generated by the AWS ML Audit Tool. It supports data residency, transparency, fairness and explainability checking._
    """
)