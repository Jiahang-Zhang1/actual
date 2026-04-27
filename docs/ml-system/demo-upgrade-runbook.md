# Demo Upgrade and CI Recovery Runbook

This runbook is the short operational path for upgrading the Smart Transaction
Categorization demo before a class demo or during the one-week freeze window.

## Golden Path

1. Confirm the application repo is clean and pushed to GitHub:

   ```bash
   cd /Users/junliu/git_repo/actual
   git status -sb
   git log --oneline -5
   gh run list --repo Jiahang-Zhang1/actual --branch master --limit 10
   ```

2. Required green GitHub Actions signals:
   - `Build`
   - `Test`
   - `CodeQL`
   - `Sparse Input Regression`
   - `Smart Transaction MLOps Automation`

3. Rebuild the Actual Budget image from the latest commit:

   ```bash
   cd /Users/junliu/git_repo/actual
   TAG=serving-$(git rev-parse --short HEAD)-latest \
   TAR_PATH=artifacts/docker/actual-smartcat_actual-sync-serving-$(git rev-parse --short HEAD)-latest-linux_amd64.tar.gz \
   bash serving/tools/build_actual_sync_image.sh tar
   ```

4. Rebuild the SmartCat serving image:

   ```bash
   cd /Users/junliu/git_repo/actual
   docker buildx build \
     --platform linux/amd64 \
     --file serving/docker/Dockerfile \
     --tag actualbudget-serving:latest \
     --output type=docker,dest=artifacts/docker/actualbudget-serving_latest-linux_amd64.tar.gz \
     serving
   ```

5. Copy both image tarballs and the DevOps repo to node1, then publish them to
   the in-cluster registry:

   ```bash
   rsync -avz -e "ssh -i ~/.ssh/id_rsa_chameleon" \
     artifacts/docker/actual-smartcat_actual-sync-serving-$(git rev-parse --short HEAD)-latest-linux_amd64.tar.gz \
     artifacts/docker/actualbudget-serving_latest-linux_amd64.tar.gz \
     cc@129.114.26.122:/home/cc/proj08-images/

   rsync -avz --delete \
     --exclude .git \
     --exclude .venv-kubespray \
     --exclude artifacts \
     -e "ssh -i ~/.ssh/id_rsa_chameleon" \
     /Users/junliu/git_repo/proj08-iac/ \
     cc@129.114.26.122:/home/cc/proj08-iac/

   ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.26.122
   cd /home/cc/proj08-iac
   IMAGE_TAR=/home/cc/proj08-images/actual-smartcat_actual-sync-serving-<sha>-latest-linux_amd64.tar.gz \
     bash scripts/load-actual-image.sh
   SERVING_IMAGE_TAR=/home/cc/proj08-images/actualbudget-serving_latest-linux_amd64.tar.gz \
     bash scripts/load-serving-image.sh
   ```

6. Roll out the latest images:

   ```bash
   sudo kubectl --kubeconfig /etc/kubernetes/admin.conf -n actual-budget rollout restart deployment/actual-sync
   sudo kubectl --kubeconfig /etc/kubernetes/admin.conf -n smartcat rollout restart deployment/smartcat-serving
   sudo kubectl --kubeconfig /etc/kubernetes/admin.conf -n actual-budget rollout status deployment/actual-sync --timeout=180s
   sudo kubectl --kubeconfig /etc/kubernetes/admin.conf -n smartcat rollout status deployment/smartcat-serving --timeout=180s
   ```

7. Run public smoke and sparse-input regression:

   ```bash
   cd /Users/junliu/git_repo/proj08-iac
   bash scripts/check-public-services.sh 129.114.26.122
   bash scripts/freeze-window-smoke.sh --floating-ip 129.114.26.122 --iterations 2 --interval 15

   cd /Users/junliu/git_repo/actual
   python3 scripts/run_manual_input_robustness.py --base-url http://129.114.26.122:30090
   python3 scripts/generate_sparse_input_matrix.py \
     --rows-per-category 2 \
     --json-output artifacts/test-data/sparse_input_matrix_demo.json \
     --csv-output artifacts/test-data/sparse_input_matrix_demo_manifest.csv
   python3 scripts/test_sparse_input_matrix.py \
     --base-url http://129.114.26.122:30090 \
     --matrix-json artifacts/test-data/sparse_input_matrix_demo.json \
     --mode both \
     --batch-size 25 \
     --traffic-source sparse-input-demo \
     --max-failures 0 \
     --output-json artifacts/test-results/sparse_input_matrix_demo_summary.json \
     --raw-output-json artifacts/test-results/sparse_input_matrix_demo_results.json
   ```

8. Open demo pages:

   ```bash
   cd /Users/junliu/git_repo/proj08-iac
   bash scripts/open-demo-pages.sh 129.114.26.122
   ```

## Recent CI Failure Root Causes

### `Test` workflow failed on formatting

Symptom:

```text
Format issues found. Run without --check to fix.
```

Fix:

```bash
yarn lint:fix
yarn lint
```

Also keep generated artifacts under ignored paths such as
`artifacts/mlops-manual-robust/`.

### `Smart Transaction MLOps Automation` failed on MLflow system metrics

Symptom:

```text
Failed to start system metrics monitoring as package psutil is not installed
```

Fix:

`training/requirements.txt` must include `psutil`.

Validation:

```bash
gh workflow run "Smart Transaction MLOps Automation" --repo Jiahang-Zhang1/actual --ref master
gh run watch <run-id> --repo Jiahang-Zhang1/actual --exit-status
```

### `Sparse Input Regression` failed waiting for `/readyz`

Symptom:

```text
FileNotFoundError: Missing ONNX model artifact: /workspace/models/optimized/...
```

Cause:

The workflow was running directly on a GitHub runner, not inside the Docker
image, so container-only model paths did not exist.

Fix:

- The sparse workflow exports `MODEL_PATH` and `SOURCE_MODEL_PATH` from the
  GitHub checkout.
- `serving/models/manifest.json` uses relative paths so it works both in Docker
  and in a normal checkout.

Validation:

```bash
gh workflow run "Sparse Input Regression" --repo Jiahang-Zhang1/actual --ref master
gh run watch <run-id> --repo Jiahang-Zhang1/actual --exit-status
```

## Demo URLs

Use HTTPS for the Actual UI:

- Actual Budget: `https://129.114.26.122:30443`
- SmartCat API docs: `http://129.114.26.122:30090/docs`
- Grafana: `http://129.114.26.122:30300/login`
- Prometheus: `http://129.114.26.122:30909`
- MLflow: `http://129.114.26.122:8000`
- MinIO Console: `http://129.114.26.122:9001`

Do not use `http://129.114.26.122:30083` as the main UI demo path. It is only
the health/fallback endpoint because Chrome blocks `SharedArrayBuffer` on public
HTTP.
