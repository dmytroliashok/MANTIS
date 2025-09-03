#!/usr/bin/env python3
"""
Quick analysis script for the specific hotkey you mentioned.
"""

import os
import sys
from analyze_embeddings import analyze_miner_embeddings, print_analysis_report
from ledger import DataLog
import config

def main():
    target_hotkey = "5DXgzL7bdDtWaokxXC6b9NSk8DPD8LKWSr1rcH1mnAZDiBgV"
    datalog_path = os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl")
    
    if not os.path.exists(datalog_path):
        print(f"❌ Datalog file not found: {datalog_path}")
        print("The validator needs to be running and have saved data first.")
        return
    
    print(f"📂 Loading datalog from: {datalog_path}")
    try:
        datalog = DataLog.load(datalog_path)
        print(f"✅ Loaded datalog with {len(datalog.blocks)} blocks")
        print(f"🔑 Total hotkeys in system: {len(datalog.live_hotkeys)}")
    except Exception as e:
        print(f"❌ Failed to load datalog: {e}")
        return
    
    print(f"\n🔍 Analyzing embeddings for: {target_hotkey}")
    analysis = analyze_miner_embeddings(
        datalog, 
        target_hotkey, 
        max_entries=10,  # Show last 10 entries per asset
        show_raw_embeddings=False  # Set to True if you want to see raw numbers
    )
    
    print_analysis_report(analysis, verbose=True)
    
    # Additional insights
    if "error" not in analysis:
        print("\n🧠 INSIGHTS")
        
        # Check which assets this miner is most active on
        asset_activity = {}
        for asset, data in analysis["assets"].items():
            activity_rate = (data["stats"]["non_zero_submissions"] / 
                           max(1, data["stats"]["total_submissions"])) * 100
            asset_activity[asset] = activity_rate
        
        # Sort by activity
        sorted_assets = sorted(asset_activity.items(), key=lambda x: x[1], reverse=True)
        print("  Most active assets:")
        for asset, rate in sorted_assets[:5]:
            print(f"    {asset}: {rate:.1f}% activity rate")
        
        # Check consistency
        btc_submissions = analysis["assets"]["BTC"]["stats"]["non_zero_submissions"]
        if btc_submissions > 5:
            print(f"  This miner has {btc_submissions} non-zero BTC submissions")
            print("  This suggests they have an active strategy")
        else:
            print("  Limited BTC activity - may be inactive or new")

if __name__ == "__main__":
    main()