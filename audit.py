from sagemaker import clarify
from sagemaker.experiments.run import Run
import datetime
import boto3
import pandas as pd
from urllib.parse import urlparse
from sagemaker.session import Session
import re
import json

class AuditClass:
    def __init__(self, input_data: dict, audit_config: dict):
        """Initialize AuditClass with input data and audit configuration."""
        self.s3 = boto3.client('s3')
        self.sagemaker = boto3.client('sagemaker')
        self.input_data = input_data
        self.audit_config = audit_config
        self.model_name = input_data['model_name']
        self.train_data_path = input_data['train_data_path']
        self.sagemaker_session = Session()
        self.execution_role = input_data.get('execution_role', None)

    def dir_exists(self, bucket, key):
        """Check if a directory exists in S3."""
        try:
            self.s3.get_object(Bucket=bucket, Key=key)
            return True
        except self.s3.exceptions.NoSuchKey:
            return False
        except Exception as e:
            # Log or handle other exceptions as needed
            return False

    def evaluate_data_residency(self):
        """Evaluate S3 bucket residency and encryption status."""
        model_bucket = urlparse(self.input_data['model_data'], allow_fragments=False).netloc
        data_bucket = urlparse(self.input_data['train_data_path'], allow_fragments=False).netloc
        buckets_list = [model_bucket, data_bucket]
        bucket_check_list = []
        overview_list = []
        suggestion_set = set()
        num_non_eu = 0
        num_unencrypted = 0

        for bucket in buckets_list:
            # Get bucket location
            location = self.s3.get_bucket_location(Bucket=bucket).get('LocationConstraint') or 'us-east-1'
            is_eu = location.startswith('eu-')
            if not is_eu:
                num_non_eu += 1

            # Check bucket encryption
            try:
                enc = self.s3.get_bucket_encryption(Bucket=bucket)
                rules = enc['ServerSideEncryptionConfiguration']['Rules']
                encryption_type = rules[0]['ApplyServerSideEncryptionByDefault']['SSEAlgorithm']
                is_encrypted = True
            except self.s3.exceptions.ClientError as e:
                encryption_type = None
                is_encrypted = False
                num_unencrypted += 1

            bucket_check_list.append({
                'bucket_name': bucket,
                'location': {'value': location, 'is_eu': is_eu},
                'encryption': {'type': encryption_type, 'is_encrypted': is_encrypted}
            })

        all_buckets_eu = num_non_eu == 0
        all_buckets_encrypted = num_unencrypted == 0
        status = all_buckets_eu and all_buckets_encrypted

        # Overview and suggestions
        if all_buckets_eu:
            overview_list.append("All buckets are located inside EU.")
        else:
            overview_list.append(f"{num_non_eu} bucket(s) are located outside EU.")
            suggestion_set.add("Prefer S3 buckets in `eu-west-1` or `eu-central-1` for GDPR alignment.")

        if all_buckets_encrypted:
            overview_list.append("All buckets are encrypted.")
        else:
            overview_list.append(f"{num_unencrypted} bucket(s) are unencrypted.")
            suggestion_set.add("Enable server-side encryption for all buckets.")

        return {
            's3_buckets': bucket_check_list,
            'status': status,
            'overview': overview_list,
            'suggestion': list(suggestion_set)
        }

    def evaluate_traceability(self):
        """
        Evaluate model traceability by checking model metadata, training data, and metrics report.
        Returns a dict with metadata, status, overview, and suggestions.
        """
        response = self.sagemaker.describe_model(ModelName=self.model_name)
        model_arn = response.get('ModelArn')
        creation_time = response.get('CreationTime')
        image_path = response['PrimaryContainer'].get('Image')
        model_url = response['PrimaryContainer'].get('ModelDataUrl')
        execution_role = response.get('ExecutionRoleArn')
        train_data_path = self.input_data.get('train_data_path')

        metadata = {
            'registry_name': self.model_name,
            'model_arn': model_arn,
            'image_path': image_path,
            'model_url': model_url
        }
        missing = []
        overview_list = []
        suggestion_list = []

        # Creation date
        if creation_time:
            metadata['creation_date'] = {'value': creation_time, 'status': True}
        else:
            missing.append('creation_date')
            metadata['creation_date'] = {'value': 'Missing', 'status': False}

        # Execution role
        if execution_role:
            metadata['who_trained'] = {'value': execution_role, 'status': True}
        else:
            missing.append('who_trained')
            metadata['who_trained'] = {'value': 'Missing', 'status': False}

        # Training data
        if train_data_path and self.file_exist(train_data_path):
            metadata['training_data'] = {'value': train_data_path, 'status': True}
        else:
            missing.append('training_data')
            metadata['training_data'] = {'value': 'Missing', 'status': False}

        # Metrics report
        model_files_dir = re.split("output", model_url)[0]
        report_path = model_files_dir + "rule-output/CreateXgboostReport/xgboost_report.ipynb"
        if report_path and self.dir_exists(report_path):
            metadata['metrics'] = {'value': report_path, 'status': True}
        else:
            missing.append('metrics')
            metadata['metrics'] = {'value': 'Missing', 'status': False}

        status = len(missing) == 0
        if status:
            overview_list.append("All required traceability information is recorded.")
        else:
            overview_list.append("Traceability information is missing.")
            overview_list.append("Compliance cannot be approved.")
            suggestion_list.append("Retain versioned logs of model metadata and training logs.")

        return {
            'model_meta_data': metadata,
            'status': status,
            'overview': overview_list,
            'suggestion': suggestion_list
        }

    def evaluate_bias_explainability(self):
        """
        Runs SageMaker Clarify bias and explainability analysis on the model and training data.
        """
        instance_type = self.audit_config['instance_type']
        instance_count = self.audit_config['instance_count']
        train_uri = self.input_data['train_data_path']
        output_path = self.audit_config['audit_path']
        label = self.input_data['label']
        headers = self.input_data['train_headers']
        data_type = self.input_data['data_type']

        clarify_processor = clarify.SageMakerClarifyProcessor(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=self.sagemaker_session
        )

        data_config = clarify.DataConfig(
            s3_data_input_path=train_uri,
            s3_output_path=output_path,
            label=label,
            headers=headers,
            dataset_type=data_type,
        )

        model_config = clarify.ModelConfig(
            model_name=self.model_name,
            instance_type=instance_type,
            instance_count=instance_count,
            accept_type=data_type,
            content_type=data_type
        )

        predictions_config = clarify.ModelPredictedLabelConfig(probability_threshold=0.5)
        facet_name = self.input_data['fairness_setting']['facet_name']
        group_name = self.input_data['fairness_setting'].get('group_name')

        bias_config = clarify.BiasConfig(
            label_values_or_threshold=[1],
            facet_name=facet_name,
            facet_values_or_threshold=[0],
            group_name=group_name if group_name else None
        )

        experiment_name = f"audit_experiment-{datetime.datetime.now():%d-%m-%Y-%H-%M-%S}"

        shap_config = clarify.SHAPConfig(
            num_samples=15,
            agg_method="mean_abs",
            save_local_shap_values=True
        )

        try:
            with Run(
                experiment_name=experiment_name,
                run_name="audit",
                sagemaker_session=self.sagemaker_session,
            ) as run:
                clarify_processor.run_bias_and_explainability(
                    data_config=data_config,
                    model_config=model_config,
                    explainability_config=shap_config,
                    bias_config=bias_config,
                    model_predicted_label_config=predictions_config,
                    pre_training_methods=[],
                    post_training_methods=["AD", "TE", "DPPL", "DAR"],
                )
        except Exception as e:
            print(f"Error running bias and explainability analysis: {e}")

    def bias_output(self):
        """
        Reads and interprets the bias analysis output from SageMaker Clarify.
        Returns a dict with bias info, status, overview, and suggestions.
        """
        facet_name = self.input_data['fairness_setting']['facet_name']
        overview_list = []
        suggestion_set = set()
        output_path = self.audit_config['audit_path']
        bias_info = {'protected_feature': facet_name}
        overall_status = True

        metric_explanations = {
            'AD': ("Model prediction accuracy for the favored and disfavored facets are equal",
                   "There is significant difference between the prediction accuracy for the favored and disfavored facets."),
            'TE': ("The ratio of false positives to false negatives between the favored and disfavored facets are equal",
                   "There is significant difference in the ratio of false positives to false negatives between the favored and disfavored facets"),
            'DPPL': ("The proportion of positive predictions between the favored facet and the disfavored facet are equal",
                     "There is significant difference in the proportion of positive predictions between the favored and disfavored facets."),
            'DAR': ("The difference in the ratios of observed positive outcomes (TP) to predicted positives (TP + FP) between the favored and disfavored facets are equal",
                    "There is significant difference in the ratios of observed positive outcomes (TP) to predicted positives (TP + FP) between the favored and disfavored facets")
        }
        mitigation_suggestion = "Apply bias mitigation (e.g., reweighing, threshold tuning, or adversarial debiasing)"

        try:
            with open(output_path + 'bias_report/analysis.json') as f:
                bias_analysis = json.load(f)
            metrics = bias_analysis["post_training_bias_metrics"]["facets"][facet_name]["metrics"]
        except Exception as e:
            return {
                'bias_info': {},
                'status': False,
                'overview': ["Could not read bias analysis output."],
                'suggestion': [str(e)]
            }

        for metric in metrics:
            metric_status = metric['value'] < 0.1
            overall_status = overall_status and metric_status
            bias_info[metric['name']] = {'value': metric['value'], 'status': metric_status}

            explanation = metric_explanations.get(metric['name'])
            if explanation:
                overview_list.append(explanation[0] if metric_status else explanation[1])
                if not metric_status:
                    suggestion_set.add(mitigation_suggestion)

        return {
            'bias_info': bias_info,
            'status': overall_status,
            'overview': overview_list,
            'suggestion': list(suggestion_set)
        }

    def explainability_output(self):
        """
        Reads and interprets the explainability analysis output from SageMaker Clarify.
        Returns a dict with explainability info, status, overview, and suggestions.
        """
        output_path = self.audit_config['audit_path']
        train_uri = self.input_data['train_data_path']
        overview_list = []
        suggestion_list = []
        global_status = False
        local_status = False
        global_values = None
        local_attributions = None
        example_x_row = None
        example_yhat = None
        local_exp_plot_legend = None
        selected_example = 1

        # Global explanations
        try:
            with open(output_path + 'explainability_report/analysis.json') as f:
                explainability_analysis = json.load(f)
            global_values = explainability_analysis["explanations"]["kernel_shap"]['0']["global_shap_values"]
            global_status = True
            overview_list.append("The model outcome can be explained by global feature attributions using kernel SHAP method.")
        except Exception:
            overview_list.append("The model is unable to explain global feature attributions.")
            suggestion_list.append("Apply explainability method.")

        # Local explanations
        try:
            local_explanations_out = pd.read_csv(output_path + "/explanations_shap/out.csv")
            feature_names = [c.replace("_label0", "") for c in local_explanations_out.columns]
            local_explanations_out.columns = feature_names

            example_x_row = pd.read_csv(train_uri, skiprows=lambda x: x not in [selected_example])
            local_attributions = local_explanations_out.iloc[selected_example]
            example_yhat = sum(local_attributions)
            local_exp_plot_legend = f"Local explanation for the example number {selected_example}"

            local_status = True
            overview_list.append("The model outcome can be explained by local feature attributions using kernel SHAP method.")
        except Exception:
            overview_list.append("The model is unable to explain local feature attributions.")
            suggestion_list.append("Apply explainability method.")

        final_status = global_status and local_status

        return {
            'exp_info': {
                'method': 'kernel_shap',
                'global_importance': global_values,
                'local_importance': {
                    'example_id': selected_example,
                    'x_values': example_x_row,
                    'y_hat': example_yhat,
                    'local_importance': local_attributions,
                    'plot_legend': local_exp_plot_legend
                }
            },
            'status': final_status,
            'overview': overview_list,
            'suggestion': suggestion_list
        }
        

    def get_results(self):
        """
        Runs all audit checks and returns a summary report dictionary.
        """
        today = datetime.date.today()

        try:
            residency_check = self.evaluate_data_residency()
        except Exception as e:
            residency_check = {'status': False, 'overview': [f"Residency check failed: {e}"], 'suggestion': []}

        try:
            traceability_check = self.evaluate_traceability()
        except Exception as e:
            traceability_check = {'status': False, 'overview': [f"Traceability check failed: {e}"], 'suggestion': []}

        try:
            self.evaluate_bias_explainability()
            bias_check = self.bias_output()
        except Exception as e:
            bias_check = {'status': False, 'overview': [f"Bias check failed: {e}"], 'suggestion': []}

        try:
            explainability_check = self.explainability_output()
        except Exception as e:
            explainability_check = {'status': False, 'overview': [f"Explainability check failed: {e}"], 'suggestion': []}

        report = {
            'model_name': self.model_name,
            'audit_date': today.isoformat(),
            'audit_log': self.audit_config['audit_path'],
            'residency_check': residency_check,
            'traceability_check': traceability_check,
            'bias_check': bias_check,
            'explainability_check': explainability_check,
            'summary': {
                'residency': residency_check.get('status', False),
                'traceability': traceability_check.get('status', False),
                'bias': bias_check.get('status', False),
                'explainability': explainability_check.get('status', False)
            }
        }
        return report

