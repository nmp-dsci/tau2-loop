# Bootstrap: run ONCE, locally, with admin credentials (aws sso login).
#
# Creates only the CI deploy role. Local state on purpose — this module is what
# the demo stack's remote state depends on, so it cannot itself live there.
#
#   cd infra/terraform/bootstrap
#   terraform init && terraform apply
#
# The GitHub OIDC *provider* already exists in this account (created by
# data-qa-agent's bootstrap) and is referenced, not recreated.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "ap-southeast-1"
}

variable "ecr_regions" {
  description = "Regions the deploy role may manage ECR repositories in (ARNs embed the region)."
  type        = list(string)
  default     = ["ap-southeast-1", "ap-southeast-2"]
}

variable "project" {
  type    = string
  default = "tau2loop"
}

variable "github_repo" {
  type    = string
  default = "nmp-dsci/tau2-loop"
}

variable "github_repo_immutable" {
  description = <<-EOT
    The repo's immutable OIDC subject prefix. Repositories created after
    GitHub's 2026 change default to `use_immutable_subject`, so the token's
    `sub` reads `repo:<owner>@<owner_id>/<repo>@<repo_id>:...` rather than
    `repo:<owner>/<repo>:...` — the sibling projects predate this and their
    trust policies match the old form. Read it from
    `gh api repos/<owner>/<repo>/actions/oidc/customization/sub`.
  EOT
  type        = string
  default     = "nmp-dsci@18507240/tau2-loop@1369686232"
}

variable "tfstate_bucket" {
  description = "Existing state bucket shared with the sibling demos."
  type        = string
  default     = "data-qa-tfstate-089783391188"
}

data "aws_caller_identity" "current" {}

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*", "repo:${var.github_repo_immutable}:*"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${var.project}-github-deploy"
  description        = "Assumed by GitHub Actions (OIDC) to deploy the tau2-loop demo."
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "TerraformStateKey"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.tfstate_bucket}", "arn:aws:s3:::${var.tfstate_bucket}/tau2-loop/*"]
  }
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid     = "EcrRepo"
    actions = ["ecr:*"]
    resources = [
      for r in var.ecr_regions :
      "arn:aws:ecr:${r}:${data.aws_caller_identity.current.account_id}:repository/${var.project}-*"
    ]
  }
  statement {
    sid       = "AppRunner"
    actions   = ["apprunner:*"]
    resources = ["*"]
  }
  statement {
    sid = "IamForServiceRoles"
    actions = [
      "iam:GetRole", "iam:CreateRole", "iam:DeleteRole", "iam:TagRole", "iam:PassRole",
      "iam:ListRolePolicies", "iam:ListAttachedRolePolicies", "iam:ListInstanceProfilesForRole",
      "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:PutRolePolicy", "iam:DeleteRolePolicy",
      "iam:GetRolePolicy", "iam:CreateServiceLinkedRole",
    ]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project}-*",
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-service-role/*",
    ]
  }
  statement {
    sid = "Observability"
    actions = [
      "cloudwatch:PutMetricAlarm", "cloudwatch:DeleteAlarms", "cloudwatch:DescribeAlarms",
      "cloudwatch:ListTagsForResource", "cloudwatch:TagResource",
      "logs:CreateLogGroup", "logs:DescribeLogGroups", "logs:PutRetentionPolicy",
      "logs:ListTagsForResource", "logs:TagResource",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "${var.project}-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

output "deploy_role_arn" {
  description = "Set this as DEPLOY_ROLE_ARN in .github/workflows/deploy-aws.yml."
  value       = aws_iam_role.github_deploy.arn
}
