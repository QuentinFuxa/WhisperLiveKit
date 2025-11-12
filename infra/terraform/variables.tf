variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
  default     = "dummy_token"
}

variable "ssh_fingerprint" {
  description = "SSH public key fingerprint used to provision the droplet"
  type        = string
  sensitive   = true
  default     = "dummy_fingerprint"
}

variable "ssh_private_key_path" {
  description = "Path to the private key used for remote provision steps"
  type        = string
  default     = "/tmp/dummy_terraform_key"
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "sgp1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "daymind"
}
