# python compare_reports.py --golden "/home/hrwang/Swing-Bench/report/" --dummy "/home/hrwang/Swing-Bench/dummy_report/"
import os
import json
import argparse
import sys
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare JSON files between two directories.')
    parser.add_argument('--golden', type=str, required=True, help='Path to golden report directory')
    parser.add_argument('--dummy', type=str, required=True, help='Path to dummy report directory')
    parser.add_argument('--output', type=str, default='diff_report.jsonl', help='Output file path')
    return parser.parse_args()

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_reports(golden_dir, dummy_dir):
    golden_files = os.listdir(golden_dir)
    dummy_files = os.listdir(dummy_dir)
    
    common_files = set(golden_files) & set(dummy_files)
    differences = []
    
    for filename in common_files:
        golden_path = os.path.join(golden_dir, filename)
        dummy_path = os.path.join(dummy_dir, filename)
        
        golden_data = load_json_file(golden_path)
        dummy_data = load_json_file(dummy_path)
        
        if not golden_data or not dummy_data:
            continue
            
        golden_jobs = {item.get('job'): item for item in golden_data.get('processed_output', [])}
        dummy_jobs = {item.get('job'): item for item in dummy_data.get('processed_output', [])}
        
        common_jobs = set(golden_jobs.keys()) & set(dummy_jobs.keys())
        
        for job in common_jobs:
            golden_job = golden_jobs[job]
            dummy_job = dummy_jobs[job]
            
            level_diff = golden_job.get('level') != dummy_job.get('level')
            result_diff = golden_job.get('stepResult') != dummy_job.get('stepResult')
            
            if level_diff or result_diff:
                diff_record = {
                    'filename': filename,
                    'job': job,
                    'has_level_diff': level_diff,
                    'has_result_diff': result_diff,
                    'golden_level': golden_job.get('level'),
                    'dummy_level': dummy_job.get('level'),
                    'golden_result': golden_job.get('stepResult'),
                    'dummy_result': dummy_job.get('stepResult'),
                    'golden_job_data': golden_job,
                    'dummy_job_data': dummy_job,
                    'comparison_time': datetime.now().isoformat()
                }
                differences.append(diff_record)
    
    return differences

def save_differences(differences, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for diff in differences:
                f.write(json.dumps(diff, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(differences)} differences to {output_path}")
    except Exception as e:
        print(f"Error saving differences to {output_path}: {e}")

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.golden):
        print(f"Error: Golden directory '{args.golden}' does not exist")
        sys.exit(1)
        
    if not os.path.isdir(args.dummy):
        print(f"Error: Dummy directory '{args.dummy}' does not exist")
        sys.exit(1)
    
    print(f"Comparing files between:\n  {args.golden}\n  {args.dummy}")
    differences = compare_reports(args.golden, args.dummy)
    
    if differences:
        save_differences(differences, args.output)
        print(f"Found {len(differences)} differences")
        
        level_diffs = sum(1 for d in differences if d['has_level_diff'])
        result_diffs = sum(1 for d in differences if d['has_result_diff'])
        print(f"  - Level differences: {level_diffs}")
        print(f"  - Result differences: {result_diffs}")
    else:
        print("No differences found")

if __name__ == "__main__":
    main()