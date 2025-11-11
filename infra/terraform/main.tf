terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.36"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

resource "digitalocean_droplet" "app" {
  image    = "debian-12-x64"
  name     = "${var.project_name}-app"
  region   = var.region
  size     = "s-1vcpu-1gb"
  ssh_keys = [var.ssh_fingerprint]
}

resource "digitalocean_redis" "cache" {
  name    = "${var.project_name}-redis"
  region  = var.region
  size    = "db-s-1vcpu-1gb"
  engine  = "redis"
  version = "7"
}

output "app_ip" {
  value = digitalocean_droplet.app.ipv4_address
}

output "redis_uri" {
  value = digitalocean_redis.cache.uri
}
