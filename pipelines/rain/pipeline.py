"""Example workflow pipeline script.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import sys

import boto3
import logging
import sagemaker
import sagemaker.session

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    FrameworkProcessor,
    ScriptProcessor
)
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

from sagemaker.sklearn.model import SKLearnModel
from sagemaker.pytorch.model import PyTorchModel
from sagemaker import PipelineModel
from sagemaker.workflow.properties import PropertyFile

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    logger.debug("Getting session")

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    logger.info("Session Obtained")
    
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="RainAuPackageGroup",
    pipeline_name="RainAuPipeline",
    base_job_prefix="RainAu",
    project_id="SageMakerProjectId"
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    logger.info("Get session and role")
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    logger.debug("Define parameters")

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    inference_instance_type = ParameterString(
        name="InferenceInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    # Training
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://rain-data-17012022/data/weatherAUS.csv"    
    )
    training_epochs = ParameterString(name="TrainingEpochs", default_value="1")

    # Validation
    # Low threshold as it is only an example
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.6) 

    sklearn_framework_version = "0.23-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_framework_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        volume_size_in_gb=15,
        base_job_name=f"{base_job_prefix}-sklearn-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    processing_step = ProcessingStep(
        name="PreprocessRainAuData",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=input_data,
            destination='/opt/ml/processing/input/')],
        outputs=[
            ProcessingOutput(output_name="train",
                             source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test",
                             source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="scaler_model",
                             source="/opt/ml/processing/preprocess"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
    )

    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/model"

    pytorch_estimator = PyTorch(
        entry_point='train.py',
        source_dir=BASE_DIR,
        instance_type=training_instance_type,
        instance_count=1,
        framework_version='1.8.0',
        py_version='py3',
        base_job_name=f"{base_job_prefix}-torch-train",
        output_path=model_path,
        hyperparameters={'epochs': training_epochs, 'batch-size': 32, 'learning-rate': 0.00009},
        role=role
    )

    step_train = TrainingStep(
        name="TrainRainModel",
        estimator=pytorch_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ) 
        }
    )

    # Uncertain on this: prefer to use a normal ScriptProcessor
    # pytorch_processor = FrameworkProcessor(
    #     PyTorch,
    #     instance_type=processing_instance_type,
    #     instance_count=1,
    #     framework_version='1.8.0',
    #     base_job_name=f"{base_job_prefix}-torch-eval",
    #     sagemaker_session=sagemaker_session,
    #     role=role,
    #     command=["python3"],
    # )
    pytorch_eval_image = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version='1.8.0',
        image_scope="training",
        py_version="py3",
        instance_type=training_instance_type
    )
    evaluation_processor = ScriptProcessor(
        role=role,
        image_uri=pytorch_eval_image,
        command=["python3"],
        instance_count=1,
        instance_type=training_instance_type
    )
    evaluation_report = PropertyFile(
        name="RainAuEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_evaluate = ProcessingStep(
        name="EvaluateRainAuModelPerformance",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # Define the location of the scaler sklearn model artifacts
    scaler_model_artifacts = "{}/model.tar.gz".format(
        processing_step.arguments["ProcessingOutputConfig"]["Outputs"][3]["S3Output"]["S3Uri"]
    )

    # load scaler model
    scaler_model = SKLearnModel(
        model_data=scaler_model_artifacts,
        role=role,
        sagemaker_session=sagemaker_session,
        entry_point=os.path.join(BASE_DIR, "preprocess.py"), # The handler functions are defined here as well
        framework_version=sklearn_framework_version, # as before
    )

    # Load the pytorch model artifacts
    pytorch_model = PyTorchModel(
        entry_point='inference.py',
        source_dir=BASE_DIR,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        framework_version='1.8.0',
        py_version='py3',
        name="rain-au-inference-model-pipeline",
        sagemaker_session=sagemaker_session # remember to always have this
    )

    # Combine them into a single model
    pipeline_model = PipelineModel(
        models=[scaler_model, pytorch_model], role=role, sagemaker_session=sagemaker_session
    )

    evaluation_s3_uri = "{}/evaluation.json".format(
        step_evaluate.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_s3_uri,
            content_type="application/json"
        )
    )

    step_register_pipeline_model = RegisterModel(
        name="RainAuPipelineModel",
        model=pipeline_model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )

    # basically this check that <left> >= <right>
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet( # Basically this reads the property file to find the necessary metric
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="regression_metrics.accuracy", # probably should change path as it is not regression
        ),
        right=accuracy_threshold
    )

    # Turn this into a pipeline step
    step_cond = ConditionStep(
        name="rain-au-accuracy-condition",
        conditions=[condition],
        if_steps=[step_register_pipeline_model],  # step_register_model, step_register_scaler,
        else_steps=[], # if the model does not pass then nothing happens
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[ # These are all the parameters defined at the beginning that can be passed and adapted
            # as the pipelines are executed
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            inference_instance_type,
            input_data,
            model_approval_status,
            training_epochs,
            accuracy_threshold,
        ],
        steps=[processing_step, step_train, step_evaluate, step_cond]
    )

    return pipeline
