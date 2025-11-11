variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "project_name" {
  description = "Base name for resources"
  type        = string
  default     = "whisper-budget"
}

variable "ssh_fingerprint" {
  description = "SSH key fingerprint authorized on DO"
  type        = string
}

variable "region" {
  description = "DigitalOcean region slug"
  type        = string
  default     = "sgp1"
}
