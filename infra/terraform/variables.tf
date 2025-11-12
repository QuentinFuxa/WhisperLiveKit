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

variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
}

variable "ssh_fingerprint" {
  description = "SSH key fingerprint"
  type        = string
}

variable "ssh_private_key_path" {
  description = "Path to the private key that matches ssh_fingerprint (for remote provisioning)"
  type        = string
}
