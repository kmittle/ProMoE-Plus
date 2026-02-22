import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from evaluator import Evaluator, FIDStatistics


def _get_ref_cache_path(ref_npz):
    """Derive a cache file path from the reference NPZ path."""
    base = os.path.splitext(ref_npz)[0]
    return base + "_stats_cache.npz"


def _load_ref_cache(cache_path):
    """Load cached reference activations and statistics."""
    print(f"Loading cached reference statistics from {cache_path} ...")
    data = np.load(cache_path)
    ref_acts = (data["pool_acts"], data["spatial_acts"])
    ref_stats = FIDStatistics(data["mu"], data["sigma"])
    ref_stats_spatial = FIDStatistics(data["mu_s"], data["sigma_s"])
    return ref_acts, ref_stats, ref_stats_spatial


def _compute_and_save_ref_cache(evaluator, ref_npz, cache_path):
    """Compute reference activations and statistics, then save to cache."""
    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(ref_npz)
    print("Computing reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_npz, ref_acts)

    print(f"Saving reference statistics cache to {cache_path} ...")
    np.savez(
        cache_path,
        pool_acts=ref_acts[0],
        spatial_acts=ref_acts[1],
        mu=ref_stats.mu,
        sigma=ref_stats.sigma,
        mu_s=ref_stats_spatial.mu,
        sigma_s=ref_stats_spatial.sigma,
    )
    return ref_acts, ref_stats, ref_stats_spatial


def evaluate_single(evaluator, image_folder, count, ref_acts, ref_stats, ref_stats_spatial):
    """Evaluate a single image folder and return results dict."""
    print(f"\n--- Evaluating: {image_folder} ---")
    print("Computing sample batch activations from image folder...")
    sample_acts = evaluator.read_activations_from_folder(image_folder, max_count=count)
    print("Computing sample batch statistics...")
    sample_stats, sample_stats_spatial = tuple(
        evaluator.compute_statistics(x) for x in sample_acts
    )

    fid = sample_stats.frechet_distance(ref_stats)
    inception_score = evaluator.compute_inception_score(sample_acts[0])
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

    results = {
        "FID": fid,
        "Inception Score": inception_score,
        "sFID": sfid,
        "Precision": prec,
        "Recall": recall,
    }

    print("Computing evaluations...")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save results to log file
    log_path = os.path.join(os.path.dirname(image_folder.rstrip(os.sep)), "eval_results.txt")
    try:
        with open(log_path, "w") as f:
            f.write(f"Sample: {image_folder}\n")
            f.write(f"Count: {count}\n\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"Results saved to: {log_path}")
    except IOError as e:
        print(f"Warning: Unable to write log file {log_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated images against a reference NPZ file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_folders", type=str, nargs="+",
                        help="Path(s) to folder(s) containing generated images.")
    parser.add_argument("-c", "--count", type=int, default=50000,
                        help="Maximum number of images to evaluate.")
    parser.add_argument("--ref-npz", type=str, default="VIRTUAL_imagenet256_labeled.npz",
                        help="Path to the reference NPZ file for evaluation.")

    args = parser.parse_args()

    # Validate folders
    valid_folders = []
    for folder in args.image_folders:
        if os.path.isdir(folder):
            valid_folders.append(folder)
        else:
            print(f"Warning: Folder '{folder}' does not exist, skipping.")

    if not valid_folders:
        print("Error: No valid image folders provided.")
        return

    if not os.path.exists(args.ref_npz):
        print(f"Error: Reference NPZ file '{args.ref_npz}' not found.")
        return

    print(f"--- Configuration ---")
    print(f"Image Folders:  {len(valid_folders)} folder(s)")
    print(f"Max Count:      {args.count}")
    print(f"Reference NPZ:  {args.ref_npz}")
    print(f"---------------------\n")

    # Single session for all evaluations
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("Warming up TensorFlow...")
    evaluator.warmup()

    # Load reference stats once
    cache_path = _get_ref_cache_path(args.ref_npz)
    if os.path.exists(cache_path):
        ref_acts, ref_stats, ref_stats_spatial = _load_ref_cache(cache_path)
    else:
        ref_acts, ref_stats, ref_stats_spatial = _compute_and_save_ref_cache(
            evaluator, args.ref_npz, cache_path
        )

    # Evaluate all folders
    for folder in valid_folders:
        evaluate_single(evaluator, folder, args.count, ref_acts, ref_stats, ref_stats_spatial)


if __name__ == "__main__":
    main()
