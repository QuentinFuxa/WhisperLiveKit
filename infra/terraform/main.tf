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

resource "null_resource" "redis_install" {
  depends_on = [digitalocean_droplet.app]

  connection {
    host        = digitalocean_droplet.app.ipv4_address
    user        = "root"
    private_key = file(var.ssh_private_key_path)
  }

  provisioner "remote-exec" {
    inline = [
      "apt update",
      "apt install -y redis-server",
      "systemctl enable redis-server --now"
    ]
  }
}

output "app_ip" {
  value = digitalocean_droplet.app.ipv4_address
}

output "redis_uri" {
  value     = "redis://127.0.0.1:6379"
  sensitive = true
}
