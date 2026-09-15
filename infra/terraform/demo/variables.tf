# These defaults ARE the live configuration; there is no committed tfvars file.

variable "region" {
  description = "Singapore: this account is at the two-App-Runner-services cap in Sydney."
  type        = string
  default     = "ap-southeast-1"
}

variable "project" {
  type    = string
  default = "tau2loop"
}
