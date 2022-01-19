"""Example workflow pipeline script for abalone pipeline.

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

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.dataset_definition.inputs import S3Input
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    FrameworkProcessor
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel

from botocore.exceptions import ClientError


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.debug("Entered script")
logger.info("Entered script")



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


# def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
#     """ Gets ECR URI from image versions
#     Args:
#         sagemaker_session: boto3 session for sagemaker client
#         image_versions: list of the image versions
#         image_name: Name of the image

#     Returns:
#         ECR URI of the image version
#     """

#     # Fetch image details to get the Base Image URI
#     for image_version in image_versions:
#         if image_version['ImageVersionStatus'] == 'CREATED':
#             image_arn = image_version["ImageVersionArn"]
#             version = image_version["Version"]
#             logger.info(f"Identified the latest image version: {image_arn}")
#             response = sagemaker_session.sagemaker_client.describe_image_version(
#                 ImageName=image_name,
#                 Version=version
#             )
#             return response['ContainerImage']
#     return None


# def resolve_ecr_uri(sagemaker_session, image_arn):
#     """Gets the ECR URI from the image name

#     Args:
#         sagemaker_session: boto3 session for sagemaker client
#         image_name: name of the image

#     Returns:
#         ECR URI of the latest image version
#     """

#     # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
#     image_name = image_arn.partition("image/")[2]
#     try:
#         # Fetch the image versions
#         next_token = ''
#         while True:
#             response = sagemaker_session.sagemaker_client.list_image_versions(
#                 ImageName=image_name,
#                 MaxResults=100,
#                 SortBy='VERSION',
#                 SortOrder='DESCENDING',
#                 NextToken=next_token
#             )
#             ecr_uri = resolve_ecr_uri_from_image_versions(
#                 sagemaker_session, response['ImageVersions'], image_name)
#             if "NextToken" in response:
#                 next_token = response["NextToken"]

#             if ecr_uri is not None:
#                 return ecr_uri

#         # Return error if no versions of the image found
#         error_message = (
#             f"No image version found for image name: {image_name}"
#         )
#         logger.error(error_message)
#         raise Exception(error_message)

#     except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
#         error_message = e.response["Error"]["Message"]
#         logger.error(error_message)
#         raise Exception(error_message)


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
        print("Get role")

        role = sagemaker.session.get_execution_role(sagemaker_session)
    print("Define parameters")
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
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://rain-data-17012022/data/weatherAUS.csv"
        # default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )

    # # processing step for feature engineering
    # try:
    #     processing_image_uri = sagemaker_session.sagemaker_client.describe_image_version(
    #         ImageName=processing_image_name)['ContainerImage']
    # except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
    #     processing_image_uri = sagemaker.image_uris.retrieve(
    #         framework="xgboost",
    #         region=region,
    #         version="1.0-1",
    #         py_version="py3",
    #         instance_type=processing_instance_type,
        # )
    print("start preprocessing")
    sklearn_processor = SKLearnProcessor(
        framework_version='0.20.0',
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        volume_size_in_gb=15,
        base_job_name=f"{base_job_prefix}-sklearn-preprocess",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
    )
    input_data = "s3://rain-data-17012022/data/weatherAUS.csv"
    step_process = ProcessingStep(
        name="PreprocessRainAuData",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train",
                             source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test",
                             source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="scaler",
                             source="/opt/ml/processing/preprocess"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/model"
    print("training")
    pytorch_estimator = PyTorch(os.path.join(BASE_DIR, 'train.py'),
                            instance_type=training_instance_type,
                            instance_count=1,
                            framework_version='1.8.0',
                            py_version='py3',
                            base_job_name=f"{base_job_prefix}-torch-train",
                            output_path=model_path,
                            hyperparameters={'epochs': 1, 'batch-size': 32, 'learning-rate': 0.00009},
                            role=role)

    # pytorch_estimator.fit({'train': 's3://my-data-bucket/path/to/my/training/data',
    #                     'test': 's3://my-data-bucket/path/to/my/test/data'})

    step_train = TrainingStep(
        name="TrainRainModel",
        estimator=pytorch_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ) # ,
            # "scaler": TrainingInput(
            #     s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
            #         "scaler"
            #     ].S3Output.S3Uri,
            # )
        }
    )


    pytorch_processor = FrameworkProcessor(
        PyTorch,
        instance_type=processing_instance_type,
        instance_count=1,
        framework_version='1.8.0',
        base_job_name=f"{base_job_prefix}-torch-eval",
        sagemaker_session=sagemaker_session,
        role=role,
        command=["python3"],
    )

    # # processing step for evaluation
    # script_eval = ScriptProcessor(
    #     image_uri=training_image_uri,
    #     command=["python3"],
    #     instance_type=processing_instance_type,
    #     instance_count=1,
    #     base_job_name=f"{base_job_prefix}/script-abalone-eval",
    #     sagemaker_session=sagemaker_session,
    #     role=role,
    # )
    evaluation_report = PropertyFile(
        name="RainAuEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateRainAuModel",
        processor=pytorch_processor,
        inputs=[
            ProcessingInput(
                source=model_path,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation",
                             source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    # try:
    #     inference_image_uri = sagemaker_session.sagemaker_client.describe_image_version(
    #         ImageName=inference_image_name)['ContainerImage']
    # except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
    #     inference_image_uri = sagemaker.image_uris.retrieve(
    #         framework="xgboost",
    #         region=region,
    #         version="1.0-1",
    #         py_version="py3",
    #         instance_type=inference_instance_type,
    #     )
    step_register = RegisterModel(
        name="RegisterRainAuModel",
        estimator=pytorch_estimator,
        # model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_data=os.path.join(model_path, "model.tar.gz"),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    # This is just a dummy condition at the moment

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy"
        ),
        right=0.5,
    )
    step_cond = ConditionStep(
        name="CheckAccRainAuEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )

    # Model needs to be saved locally in /opt/ml/processing/model/model.tar.gz and loaded by the evaluate script 
    # on startup. Need to create a model_folder subfolder and puts the model.pth artifact inside. Puts inside a code/
    # folder with inference.py serving script and a requirements folder. Read more details at:
    # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#the-sagemaker-pytorch-model-server
    # READ WHOLE PAGE

    # model.tar.gz/
    #     |- model.pth
    #     |- code/
    #         |- inference.py
    #         |- requirements.txt 

    # Need a registration step that gets the saved pytorch model tarball and then does puts it in registry.
    # TODO: REMOVE JIT LEAVE AS NORMAL MODEL
    print("done")
    return pipeline
