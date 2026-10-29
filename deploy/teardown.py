"""Removes what deploy.py created.

    python deploy/teardown.py          # endpoint, endpoint configs and models only (stops all inference cost)
    python deploy/teardown.py --all    # also ECR repo, CodeBuild project, IAM roles and the S3 bucket
"""
from __future__ import annotations

import argparse

import boto3
import yaml

from deploy import BUILD_PROJECT, CODEBUILD_ROLE, ROOT, SAGEMAKER_ROLE, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    region, project = cfg["aws"]["region"], cfg["aws"]["project"]
    session = boto3.Session(region_name=region)
    sm = session.client("sagemaker")

    try:
        sm.delete_endpoint(EndpointName=cfg["aws"]["endpoint_name"])
        log("deleted endpoint")
    except sm.exceptions.ClientError as e:
        log(f"endpoint: {e.response['Error']['Message']}")
    # Only names this pipeline generates ("<project>-rf-<timestamp>[-<image tag>]"); never anything merely
    # containing the project name, because the account may hold other models (e.g. an older "fraud-rf-v2").
    mine = f"{project}-rf-"
    for page in sm.get_paginator("list_endpoint_configs").paginate(NameContains=project):
        for c in page["EndpointConfigs"]:
            if c["EndpointConfigName"].startswith(mine):
                sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
                log(f"deleted endpoint config {c['EndpointConfigName']}")
    for page in sm.get_paginator("list_models").paginate(NameContains=project):
        for m in page["Models"]:
            if m["ModelName"].startswith(mine):
                sm.delete_model(ModelName=m["ModelName"])
                log(f"deleted model {m['ModelName']}")
    if not args.all:
        return

    account = session.client("sts").get_caller_identity()["Account"]
    bucket = f"fraud-pipeline-{account}"
    steps = [
        ("ECR repository", lambda: session.client("ecr").delete_repository(repositoryName="fraud-rf", force=True)),
        ("CodeBuild project", lambda: session.client("codebuild").delete_project(name=BUILD_PROJECT)),
    ]
    iam = session.client("iam")
    for role in (SAGEMAKER_ROLE, CODEBUILD_ROLE):
        steps.append((f"IAM role {role}", lambda r=role: (iam.delete_role_policy(RoleName=r, PolicyName="inline"),
                                                           iam.delete_role(RoleName=r))))

    def empty_and_delete_bucket():
        b = session.resource("s3").Bucket(bucket)
        b.objects.all().delete()
        b.delete()
    steps.append((f"S3 bucket {bucket}", empty_and_delete_bucket))
    for label, fn in steps:
        try:
            fn()
            log(f"deleted {label}")
        except Exception as e:  # noqa: BLE001 - keep going so one missing resource doesn't block the rest
            log(f"{label}: {e}")


if __name__ == "__main__":
    main()
