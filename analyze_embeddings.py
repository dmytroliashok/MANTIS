#!/usr/bin/env python3
"""
MANTIS Embedding Analyzer

This script loads the validator's datalog and extracts decrypted embeddings
for a specific miner hotkey, displaying their submission history and patterns.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

import config
from ledger import DataLog


def format_embedding_stats(embedding: List[float], asset: str) -> Dict:
    """Calculate basic statistics for an embedding vector."""
    if not embedding:
        return {"empty": True}
    
    arr = np.array(embedding)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "non_zero_count": int(np.count_nonzero(arr)),
        "dimension": len(embedding),
        "l2_norm": float(np.linalg.norm(arr))
    }


def analyze_miner_embeddings(
    datalog: DataLog, 
    target_hotkey: str,
    max_entries: Optional[int] = None,
    show_raw_embeddings: bool = False
) -> Dict:
    """Extract and analyze embeddings for a specific miner."""
    
    if target_hotkey not in datalog.hk2idx:
        return {
            "error": f"Hotkey {target_hotkey} not found in datalog",
            "available_hotkeys": list(datalog.hk2idx.keys())[:10]  # Show first 10
        }
    
    hotkey_idx = datalog.hk2idx[target_hotkey]
    
    results = {
        "hotkey": target_hotkey,
        "hotkey_index": hotkey_idx,
        "total_blocks": len(datalog.blocks),
        "assets": {},
        "summary": {}
    }
    
    total_submissions = 0
    total_non_zero_submissions = 0
    
    for asset in config.ASSETS:
        asset_dim = config.ASSET_EMBEDDING_DIMS[asset]
        challenge_data = datalog.datasets[asset].challenges[0]
        
        asset_results = {
            "dimension": asset_dim,
            "submissions": [],
            "stats": {
                "total_submissions": 0,
                "non_zero_submissions": 0,
                "avg_l2_norm": 0.0,
                "submission_blocks": []
            }
        }
        
        # Iterate through all stored embeddings for this asset
        for sidx, embedding_tensor in challenge_data.emb_sparse.items():
            if hotkey_idx < embedding_tensor.shape[0]:
                embedding = embedding_tensor[hotkey_idx, :].tolist()
                
                # Calculate corresponding block number
                block_num = sidx * config.SAMPLE_EVERY
                
                # Find the actual block index in datalog.blocks
                block_idx = None
                for i, b in enumerate(datalog.blocks):
                    if b == block_num:
                        block_idx = i
                        break
                
                if block_idx is None:
                    continue
                
                # Get price at that time if available
                price = None
                if block_idx < len(datalog.asset_prices):
                    price = datalog.asset_prices[block_idx].get(asset)
                
                embedding_stats = format_embedding_stats(embedding, asset)
                is_non_zero = embedding_stats.get("non_zero_count", 0) > 0
                
                submission_entry = {
                    "block": block_num,
                    "block_index": block_idx,
                    "price": price,
                    "stats": embedding_stats,
                    "is_non_zero": is_non_zero
                }
                
                if show_raw_embeddings:
                    submission_entry["raw_embedding"] = embedding
                
                asset_results["submissions"].append(submission_entry)
                asset_results["stats"]["total_submissions"] += 1
                asset_results["stats"]["submission_blocks"].append(block_num)
                
                if is_non_zero:
                    asset_results["stats"]["non_zero_submissions"] += 1
                    asset_results["stats"]["avg_l2_norm"] += embedding_stats["l2_norm"]
        
        # Calculate averages
        if asset_results["stats"]["non_zero_submissions"] > 0:
            asset_results["stats"]["avg_l2_norm"] /= asset_results["stats"]["non_zero_submissions"]
        
        # Sort submissions by block number
        asset_results["submissions"].sort(key=lambda x: x["block"])
        
        # Apply max_entries limit if specified
        if max_entries and len(asset_results["submissions"]) > max_entries:
            asset_results["submissions"] = asset_results["submissions"][-max_entries:]
        
        results["assets"][asset] = asset_results
        total_submissions += asset_results["stats"]["total_submissions"]
        total_non_zero_submissions += asset_results["stats"]["non_zero_submissions"]
    
    # Overall summary
    results["summary"] = {
        "total_submissions_across_assets": total_submissions,
        "total_non_zero_submissions": total_non_zero_submissions,
        "activity_rate": (total_non_zero_submissions / max(1, total_submissions)) * 100,
        "first_submission_block": min(
            (min(asset["stats"]["submission_blocks"]) for asset in results["assets"].values() 
             if asset["stats"]["submission_blocks"]), 
            default=None
        ),
        "last_submission_block": max(
            (max(asset["stats"]["submission_blocks"]) for asset in results["assets"].values() 
             if asset["stats"]["submission_blocks"]), 
            default=None
        )
    }
    
    return results


def print_analysis_report(analysis: Dict, verbose: bool = False):
    """Print a formatted analysis report."""
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        if "available_hotkeys" in analysis:
            print(f"\nFirst 10 available hotkeys:")
            for hk in analysis["available_hotkeys"]:
                print(f"  {hk}")
        return
    
    print(f"🔍 Analysis for Hotkey: {analysis['hotkey']}")
    print(f"📊 Hotkey Index: {analysis['hotkey_index']}")
    print(f"📈 Total Blocks in Datalog: {analysis['total_blocks']}")
    print()
    
    summary = analysis["summary"]
    print("📋 SUMMARY")
    print(f"  Total Submissions: {summary['total_submissions_across_assets']}")
    print(f"  Non-Zero Submissions: {summary['total_non_zero_submissions']}")
    print(f"  Activity Rate: {summary['activity_rate']:.1f}%")
    
    if summary['first_submission_block']:
        print(f"  First Submission: Block {summary['first_submission_block']}")
    if summary['last_submission_block']:
        print(f"  Last Submission: Block {summary['last_submission_block']}")
    print()
    
    print("💰 PER-ASSET BREAKDOWN")
    for asset, data in analysis["assets"].items():
        stats = data["stats"]
        print(f"  {asset} (dim={data['dimension']}):")
        print(f"    Submissions: {stats['total_submissions']} total, {stats['non_zero_submissions']} non-zero")
        
        if stats['non_zero_submissions'] > 0:
            print(f"    Avg L2 Norm: {stats['avg_l2_norm']:.4f}")
        
        if verbose and data["submissions"]:
            print(f"    Recent submissions:")
            recent = data["submissions"][-3:]  # Show last 3
            for sub in recent:
                status = "✅ Active" if sub["is_non_zero"] else "⭕ Zero"
                price_info = f"${sub['price']:.2f}" if sub['price'] else "No price"
                print(f"      Block {sub['block']}: {status} | {price_info} | L2={sub['stats'].get('l2_norm', 0):.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze miner embeddings from MANTIS datalog")
    parser.add_argument(
        "hotkey", 
        help="The miner's hotkey to analyze"
    )
    parser.add_argument(
        "--datalog-path", 
        default=os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl"),
        help="Path to the datalog file"
    )
    parser.add_argument(
        "--max-entries", 
        type=int, 
        help="Maximum number of recent entries to show per asset"
    )
    parser.add_argument(
        "--show-raw", 
        action="store_true",
        help="Include raw embedding vectors in output"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed submission history"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Output results as JSON instead of formatted text"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.datalog_path):
        print(f"❌ Datalog file not found: {args.datalog_path}")
        print("Make sure the validator has been running and has saved data.")
        sys.exit(1)
    
    print(f"📂 Loading datalog from: {args.datalog_path}")
    try:
        datalog = DataLog.load(args.datalog_path)
        print(f"✅ Loaded datalog with {len(datalog.blocks)} blocks")
    except Exception as e:
        print(f"❌ Failed to load datalog: {e}")
        sys.exit(1)
    
    print(f"🔍 Analyzing embeddings for: {args.hotkey}")
    analysis = analyze_miner_embeddings(
        datalog, 
        args.hotkey, 
        max_entries=args.max_entries,
        show_raw_embeddings=args.show_raw
    )
    
    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_analysis_report(analysis, verbose=args.verbose)


if __name__ == "__main__":
    main()