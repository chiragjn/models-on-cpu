import argparse
import huggingface_hub

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()


huggingface_hub.snapshot_download(
    repo_id=args.model_id,
    local_dir=args.output_dir,
    cache_dir=None,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.ot",
                     "pytorch_model*.bin", "*.h5", "*.tflite"],
)
