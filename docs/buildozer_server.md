# Server-Side Buildozer Pipeline

Heavy Buildozer runs (Android SDK/NDK downloads + APK linking) can saturate a laptop for several minutes.  
Reuse the existing Terraform droplet to offload this work and keep the Python/Kivy client reproducible.

## 1. Provision a Builder Droplet
```bash
cd infra/terraform
terraform init
terraform apply \
  -var="project_name=daymind-buildozer" \
  -var="region=fra1" \
  -var="do_token=$DO_TOKEN" \
  -var="ssh_fingerprint=$SSH_FINGERPRINT" \
  -var="ssh_private_key_path=$HOME/.ssh/id_rsa"
```

- Capture the `app_ip` output; Terraform already opens ports 22/8000 only.
- Redis provisioning from this module can remain enabled—the droplet doubles as a CI smoke target if needed.

## 2. Sync the Repo
```bash
rsync -az --delete . root@${APP_IP}:/opt/daymind
ssh root@${APP_IP}
```

All build commands below run on the droplet.

## 3. Prepare Buildozer Dependencies
From `/opt/daymind`:
```bash
scripts/setup_buildozer_host.sh
source mobile/daymind/.venv-buildozer/bin/activate
```

The script installs the Buildozer toolchain (OpenJDK 17, SDK/NDK helpers, Cython) without touching Terraform or Docker assets.

## 4. Build the APK
```bash
cd mobile/daymind
buildozer -v android debug
# or: scripts/build_apk.sh  (runs the same command and copies dist/daymind-debug.apk)
```

Artifacts live under `mobile/daymind/bin/` and are mirrored to `dist/daymind-debug.apk`.  
Use `scp root@${APP_IP}:/opt/daymind/dist/daymind-debug.apk ./dist/` to pull the file locally.

## 5. Clean Up
- Deactivate the virtualenv when done: `deactivate`.
- Tear down the droplet to save cost: `terraform destroy -var="..."`.

## Notes
- The Buildozer spec lives at `mobile/daymind/buildozer.spec`; it sets the necessary Android permissions (`RECORD_AUDIO`, `INTERNET`, `WAKE_LOCK`, `FOREGROUND_SERVICE`).
- Builds can continue locally via `scripts/build_apk.sh` if you already have Buildozer configured.
- Terraform, Docker, and the backend stack stay untouched; this workflow only adds an optional builder target.
