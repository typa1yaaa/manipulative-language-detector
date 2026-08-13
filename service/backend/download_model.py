from huggingface_hub import snapshot_download

from src.training import config


if __name__ == "__main__":
    print(f"Downloading model: {config.HF_MODEL_REPO_ID}")

    path = snapshot_download(
        repo_id=config.HF_MODEL_REPO_ID,
    )

    print(f"MODEL DOWNLOAD COMPLETE: {path}")