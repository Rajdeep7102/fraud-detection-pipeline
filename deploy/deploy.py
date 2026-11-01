"""Deploys the trained model to a SageMaker serverless endpoint. Idempotent: safe to re-run.

Steps: S3 bucket -> model.tar.gz upload -> IAM roles -> ECR repo -> CodeBuild builds the serving image
(the local Docker daemon is not needed) -> SageMaker model + serverless endpoint config -> create or update
the endpoint -> smoke test.

    python deploy/deploy.py [--skip-build]       # --skip-build reuses the image tag from the last deploy
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "deploy" / "last_deploy.json"
SERVING_FILES = ["Dockerfile", "requirements.txt", "audit.py", "scoring.py", "app.py", "serve"]
MODEL_FILES = ["model.joblib", "model_meta.json", "feature_spec.yaml", "baseline.json"]


def model_dir() -> Path:
    """artifacts/ if you trained a model, else the ready-made models/demo/ (same rule as fraud.config.model_dir)."""
    for d in (ROOT / "artifacts", ROOT / "models" / "demo"):
        if (d / "model.joblib").exists():
            return d
    raise SystemExit("no model found: run python -m fraud.train.train, or use the bundled models/demo/")

SAGEMAKER_ROLE = "FraudSageMakerExecutionRole"
CODEBUILD_ROLE = "FraudCodeBuildRole"
BUILD_PROJECT = "fraud-rf-image-build"

BUILDSPEC = """version: 0.2
phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
  build:
    commands:
      - docker build -t $IMAGE_URI .
  post_build:
    commands:
      - docker push $IMAGE_URI
"""


def load_cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- S3
def ensure_bucket(s3, name: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=name)
    except ClientError:
        kwargs = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=name, **kwargs)
        log(f"created bucket {name}")
    s3.put_public_access_block(Bucket=name, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})


def upload_model(s3, bucket: str, version: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in MODEL_FILES:
            tar.add(model_dir() / f, arcname=f)
    key = f"models/{version}/model.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    log(f"uploaded s3://{bucket}/{key} ({len(buf.getvalue()) / 1e6:.0f} MB)")
    return f"s3://{bucket}/{key}"


# --------------------------------------------------------------------------------------------- IAM
def ensure_role(iam, name: str, service: str, policy: dict) -> str:
    trust = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}
    created = False
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="Fraud detection pipeline")["Role"]["Arn"]
        log(f"created role {name}")
        created = True
    iam.put_role_policy(RoleName=name, PolicyName="inline", PolicyDocument=json.dumps(policy))
    if created:
        time.sleep(20)   # IAM is eventually consistent: wait for the role AND its policy to propagate
    return arn


def sagemaker_policy(bucket: str, account: str, region: str) -> dict:
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": f"arn:aws:s3:::{bucket}/*"},
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
        {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
         "Resource": f"arn:aws:ecr:{region}:{account}:repository/fraud-rf"},
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                                       "logs:DescribeLogStreams"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": "*"}]}


def codebuild_policy(bucket: str, repo_arn: str) -> dict:
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"], "Resource": f"arn:aws:s3:::{bucket}/build/*"},
        {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                                       "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage",
                                       "ecr:GetDownloadUrlForLayer"], "Resource": repo_arn},
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "*"}]}


# --------------------------------------------------------------------------------- image (CodeBuild)
def serving_tag() -> str:
    h = hashlib.sha256()
    for f in SERVING_FILES:
        h.update(f.encode())
        h.update((ROOT / "serving" / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def ensure_repo(ecr, name: str) -> str:
    try:
        return ecr.describe_repositories(repositoryNames=[name])["repositories"][0]["repositoryArn"]
    except ecr.exceptions.RepositoryNotFoundException:
        repo = ecr.create_repository(repositoryName=name, imageScanningConfiguration={"scanOnPush": True})
        log(f"created ECR repository {name}")
        return repo["repository"]["repositoryArn"]


def image_exists(ecr, repo: str, tag: str) -> bool:
    try:
        ecr.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])
        return True
    except ClientError:
        return False


def build_image(s3, cb, role_arn: str, bucket: str, image_uri: str, registry: str, region: str, tag: str) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SERVING_FILES:
            z.writestr(f, (ROOT / "serving" / f).read_bytes().replace(b"\r\n", b"\n"))
    key = f"build/serving-{tag}.zip"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

    params = dict(
        source={"type": "S3", "location": f"{bucket}/{key}", "buildspec": BUILDSPEC},
        artifacts={"type": "NO_ARTIFACTS"},
        environment={"type": "LINUX_CONTAINER", "image": "aws/codebuild/standard:7.0",
                     "computeType": "BUILD_GENERAL1_SMALL", "privilegedMode": True},
        serviceRole=role_arn, timeoutInMinutes=20)
    try:
        cb.create_project(name=BUILD_PROJECT, **params)
    except cb.exceptions.ResourceAlreadyExistsException:
        cb.update_project(name=BUILD_PROJECT, **params)

    build = cb.start_build(projectName=BUILD_PROJECT, sourceLocationOverride=f"{bucket}/{key}", environmentVariablesOverride=[
        {"name": "IMAGE_URI", "value": image_uri}, {"name": "ECR_REGISTRY", "value": registry},
        {"name": "AWS_REGION", "value": region}])["build"]["id"]
    log(f"CodeBuild {build} started")
    while True:
        time.sleep(10)
        b = cb.batch_get_builds(ids=[build])["builds"][0]
        phase = b.get("currentPhase")
        if b["buildStatus"] != "IN_PROGRESS":
            break
        print(f"  ... {phase}", flush=True)
    if b["buildStatus"] != "SUCCEEDED":
        for p in b.get("phases", []):
            if p.get("phaseStatus") not in (None, "SUCCEEDED"):
                log(f"{p['phaseType']} {p['phaseStatus']}: {p.get('contexts')}")
        raise SystemExit(f"image build {b['buildStatus']}; logs: {b.get('logs', {}).get('deepLink')}")
    log(f"image pushed: {image_uri}")


# --------------------------------------------------------------------------------------- SageMaker
def ensure_audit_group(logs, group: str, retention_days: int = 30) -> None:
    try:
        logs.create_log_group(logGroupName=group)
        log(f"created audit log group {group}")
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    logs.put_retention_policy(logGroupName=group, retentionInDays=retention_days)


def deploy_endpoint(sm, cfg: dict, role_arn: str, image_uri: str, model_data: str, version: str, tag: str) -> None:
    aws = cfg["aws"]
    # The name carries the image tag: a SageMaker model is immutable, so a new image or environment needs a new model.
    name = f"{aws['project']}-{version}-{tag}"
    for attempt in range(6):
        try:
            sm.create_model(ModelName=name, ExecutionRoleArn=role_arn,
                            PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_data,
                                              "Environment": {"MODEL_DIR": "/opt/ml/model", "AUDIT_LOG_GROUP": aws["audit_log_group"],
                                                              "AWS_DEFAULT_REGION": aws["region"]}})
            break
        except sm.exceptions.ClientError as e:
            msg = str(e)
            if "already exists" in msg:
                break
            if attempt == 5 or "assume" not in msg.lower():
                raise
            time.sleep(10)   # role not propagated yet
    cfg_name = f"{name}-cfg"
    try:
        sm.create_endpoint_config(EndpointConfigName=cfg_name, ProductionVariants=[{
            "VariantName": "AllTraffic", "ModelName": name,
            "ServerlessConfig": {"MemorySizeInMB": aws["serverless_memory_mb"],
                                 "MaxConcurrency": aws["serverless_max_concurrency"]}}])
    except sm.exceptions.ClientError as e:
        if "already exists" not in str(e):
            raise
    ep = aws["endpoint_name"]
    try:
        sm.describe_endpoint(EndpointName=ep)
        sm.update_endpoint(EndpointName=ep, EndpointConfigName=cfg_name)
        log(f"updating endpoint {ep} -> {cfg_name}")
    except sm.exceptions.ClientError as e:
        if "Could not find endpoint" not in str(e):
            raise
        sm.create_endpoint(EndpointName=ep, EndpointConfigName=cfg_name)
        log(f"creating endpoint {ep}")
    t0 = time.time()
    while True:
        d = sm.describe_endpoint(EndpointName=ep)
        if d["EndpointStatus"] in ("InService", "Failed"):
            break
        time.sleep(15)
    if d["EndpointStatus"] != "InService":
        raise SystemExit(f"endpoint failed: {d.get('FailureReason')}")
    log(f"endpoint InService after {time.time() - t0:.0f}s (config {cfg_name})")


def smoke_test(runtime, endpoint: str) -> dict:
    normal = {"transaction_id": 1, "txn_count_1h": 1, "txn_count_24h": 2, "amount_sum_24h": 60.0,
              "secs_since_last_txn": 40000.0, "distinct_merchants_24h": 2, "distinct_countries_7d": 1,
              "amount_to_avg_30d": 1.0, "amount_zscore_30d": 0.1, "is_new_merchant": 0, "is_new_device": 0,
              "is_foreign_country": 0, "hour_deviation": 1.0}
    burst = {**normal, "transaction_id": 2, "txn_count_1h": 14, "txn_count_24h": 15, "amount_sum_24h": 90.0,
             "secs_since_last_txn": 45.0, "distinct_merchants_24h": 13, "is_new_merchant": 1, "is_foreign_country": 1}
    r = runtime.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                                Body=json.dumps({"instances": [normal, burst]}))
    out = json.loads(r["Body"].read())
    log("smoke test: " + json.dumps(out["predictions"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    region = cfg["aws"]["region"]
    session = boto3.Session(region_name=region)
    account = session.client("sts").get_caller_identity()["Account"]
    bucket = f"fraud-pipeline-{account}"
    meta = json.loads((model_dir() / "model_meta.json").read_text())
    version = meta["model_version"]

    s3, iam, ecr = session.client("s3"), session.client("iam"), session.client("ecr")
    cb, sm, rt = session.client("codebuild"), session.client("sagemaker"), session.client("sagemaker-runtime")
    ensure_audit_group(session.client("logs"), cfg["aws"]["audit_log_group"])

    ensure_bucket(s3, bucket, region)
    model_data = upload_model(s3, bucket, version)
    repo_arn = ensure_repo(ecr, "fraud-rf")
    sm_role = ensure_role(iam, SAGEMAKER_ROLE, "sagemaker.amazonaws.com", sagemaker_policy(bucket, account, region))
    cb_role = ensure_role(iam, CODEBUILD_ROLE, "codebuild.amazonaws.com", codebuild_policy(bucket, repo_arn))

    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    tag = serving_tag()
    image_uri = f"{registry}/fraud-rf:{tag}"
    if image_exists(ecr, "fraud-rf", tag):
        log(f"image {tag} already in ECR; skipping build")
    elif args.skip_build:
        raise SystemExit(f"--skip-build but image {tag} is not in ECR")
    else:
        build_image(s3, cb, cb_role, bucket, image_uri, registry, region, tag)

    deploy_endpoint(sm, cfg, sm_role, image_uri, model_data, version, tag)
    result = smoke_test(rt, cfg["aws"]["endpoint_name"])
    STATE.write_text(json.dumps({"endpoint": cfg["aws"]["endpoint_name"], "model_version": version, "image": image_uri,
                                 "model_data": model_data, "bucket": bucket, "region": region}, indent=2))
    log(f"done; state in {STATE.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()
