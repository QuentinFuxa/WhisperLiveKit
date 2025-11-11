variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "daymind"
}

variable "ssh_fingerprint" {
  description = "SSH key fingerprint"
  type        = string
}

variable "ssh_private_key_path" {
  description = "Path to the private key that matches ssh_fingerprint (for remote provisioning)"
  type        = string
}

variable "region" {
  description = "DigitalOcean region slug"
  type        = string
  default     = "sgp1"
}
