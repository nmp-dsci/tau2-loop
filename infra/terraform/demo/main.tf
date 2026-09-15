# The demo stack: one ECR repo, one App Runner service. No VPC, no database,
# no Secrets Manager — the container is read-only by construction (DEMO_MODE is
# baked into the image) and holds no keys. Copied from ConvFinQA-agent.
#
#   terraform init
#   terraform apply -target=aws_ecr_repository.demo   # repo before first push
#   ../../../scripts/aws_build_push.sh                 # image must exist first
#   terraform apply                                    # then the service starts

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  backend "s3" {
    bucket       = "data-qa-tfstate-089783391188"
    key          = "tau2-loop/demo.tfstate"
    region       = "ap-southeast-2"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}

resource "aws_ecr_repository" "demo" {
  name                 = "${var.project}-demo"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "demo" {
  repository = aws_ecr_repository.demo.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the 5 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

data "aws_iam_policy_document" "apprunner_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_ecr_access" {
  name               = "${var.project}-apprunner-ecr-access"
  assume_role_policy = data.aws_iam_policy_document.apprunner_trust.json
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_service" "demo" {
  service_name = "${var.project}-demo"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }
    auto_deployments_enabled = true
    image_repository {
      image_identifier      = "${aws_ecr_repository.demo.repository_url}:latest"
      image_repository_type = "ECR"
      image_configuration {
        port = "8080"
      }
    }
  }

  instance_configuration {
    cpu    = "512"
    memory = "1024"
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/healthz"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  observability_configuration {
    observability_enabled = false
  }
}

resource "aws_cloudwatch_metric_alarm" "http_5xx" {
  alarm_name          = "${var.project}-demo-5xx"
  alarm_description   = "Sustained 5xx from the tau2-loop demo."
  namespace           = "AWS/AppRunner"
  metric_name         = "5xxStatusResponses"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    ServiceName = aws_apprunner_service.demo.service_name
    ServiceID   = aws_apprunner_service.demo.service_id
  }
}

output "service_url" {
  value = "https://${aws_apprunner_service.demo.service_url}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.demo.repository_url
}
