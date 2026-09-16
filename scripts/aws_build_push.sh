#!/usr/bin/env bash
# Build the demo image for linux/amd64 and push :latest to ECR. App Runner
# auto-deploys on push, so this IS the release step. CI runs the same script;
# locally it needs a live SSO session (aws sso login --profile data-qa).
set -euo pipefail

PROFILE_ARGS=()
if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  PROFILE_ARGS=(--profile "${AWS_PROFILE:-data-qa}")
fi
REGION="${AWS_REGION:-ap-southeast-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text "${PROFILE_ARGS[@]}")"
REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/tau2loop-demo"
SHA="$(git rev-parse --short HEAD)"

aws ecr get-login-password --region "$REGION" "${PROFILE_ARGS[@]}" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

docker buildx build \
  --platform linux/amd64 \
  --build-arg CODE_SHA="$SHA" \
  --tag "${REPO}:latest" \
  --tag "${REPO}:${SHA}" \
  --push \
  .

echo "pushed ${REPO}:latest (${SHA})"
