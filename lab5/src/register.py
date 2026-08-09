import os
import json

from huggingface_hub import HfApi, create_repo


def build_model_card(metrics):
    lines = [
        "---",
        "tags: [sklearn, random-forest, mlops-pipeline, boston]",
        "---",
        "# Boston Housing Classification Model",
        "",
        "Trained automatically using a Random Forest classifier.",
        "",
        "The Boston Housing dataset is converted into a binary classification problem.",
        "",
        "## Metrics",
        "",
    ]

    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value:.4f}")

    return "\n".join(lines)


def main():

    # Get credentials from environment variables
    repo_id = os.environ["HF_REPO_ID"]
    hf_token = os.environ["HF_TOKEN"]

    # Read evaluation metrics
    with open("metrics.json") as f:
        metrics = json.load(f)

    # Connect to Hugging Face
    api = HfApi(token=hf_token)

    # Create repository if it doesn't exist
    create_repo(
        repo_id,
        token=hf_token,
        exist_ok=True
    )

    # Create model card
    with open("model/README.md", "w") as f:
        f.write(build_model_card(metrics))

    # Upload model files
    files_to_upload = [
        "model/model.joblib",
        "model/features.json",
        "model/README.md"
    ]

    for path in files_to_upload:

        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=repo_id,
            token=hf_token
        )

    print(
        f"Model pushed to "
        f"https://huggingface.co/{repo_id}"
    )


if __name__ == "__main__":
    main()
