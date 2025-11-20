#!/usr/bin/env python3
"""
Extract verification results from MIP final.json files

Extracts: id, removed_conditions, incomplete_question,
          correctness_passed, correctness_analysis,
          validity_passed, validity_analysis
"""
import json
import csv
import sys
import argparse
from pathlib import Path

def extract_verification_data(input_file, output_file=None, output_format='json'):
    """
    Extract verification data from final.json file

    Args:
        input_file: Path to input JSON file
        output_file: Path to output file (default: input_file_extracted.csv/json)
        output_format: 'csv' or 'json'
    """
    # Read input JSON
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract fields
    extracted_data = []
    for item in data:
        original_id = item.get('id', 'unknown')
        # 修正：字段名改为 'original_question'
        original_question = item.get('original_question', '')

        # Process each removal variant
        variants = item.get('removal_variants', [])
        for variant in variants:
            variant_id = variant.get('variant_id', original_id)

            # Extract removed_conditions (list) - join with semicolon
            removed_conditions = variant.get('removed_conditions', [])
            if isinstance(removed_conditions, list):
                removed_conditions_str = '; '.join(removed_conditions)
            else:
                removed_conditions_str = str(removed_conditions)

            # Extract incomplete_question
            incomplete_question = variant.get('incomplete_question', '')

            # Extract verification results
            verification = variant.get('verification', {})

            # 修正：从 round_A 中获取 correctness 和 validity 信息
            round_a = verification.get('round_A', {})
            correctness_passed = round_a.get('correctness_passed', None)
            correctness_analysis = round_a.get('correctness_analysis', '')
            validity_passed = round_a.get('validity_passed', None)
            validity_analysis = round_a.get('validity_analysis', '')

            # 修正：字段名改为大写 A、B
            round_a_passed = verification.get('round_A_passed', None)
            round_b_passed = verification.get('round_B_passed', None)
            is_valid = verification.get('is_valid', None)

            extracted_data.append({
                'id': variant_id,
                'original_id': original_id,
                'original_question': original_question,
                'removed_conditions': removed_conditions_str,
                'incomplete_question': incomplete_question,
                'correctness_passed': correctness_passed,
                'correctness_analysis': correctness_analysis,
                'validity_passed': validity_passed,
                'validity_analysis': validity_analysis,
                'round_a_passed': round_a_passed,
                'round_b_passed': round_b_passed,
                'is_valid': is_valid
            })

    print(f"Extracted {len(extracted_data)} variants from {len(data)} original problems")

    # Determine output file
    if output_file is None:
        input_path = Path(input_file)
        if output_format == 'json':
            output_file = input_path.parent / f"{input_path.stem}_extracted.json"
        else:
            output_file = input_path.parent / f"{input_path.stem}_extracted.csv"

    # Write output
    if output_format == 'json':
        print(f"Writing JSON to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    else:
        print(f"Writing CSV to {output_file}...")
        fieldnames = [
            'id', 'original_id', 'original_question', 'removed_conditions', 'incomplete_question',
            'correctness_passed', 'correctness_analysis',
            'validity_passed', 'validity_analysis',
            'round_a_passed', 'round_b_passed', 'is_valid'
        ]
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(extracted_data)

    print(f"✓ Done! Output saved to: {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    total = len(extracted_data)

    if total > 0:
        correctness_passed_count = sum(1 for d in extracted_data if d.get('correctness_passed') == True)
        validity_passed_count = sum(1 for d in extracted_data if d.get('validity_passed') == True)

        print(f"Total variants: {total}")
        print(f"Correctness passed: {correctness_passed_count} ({correctness_passed_count/total*100:.1f}%)")
        print(f"Validity passed: {validity_passed_count} ({validity_passed_count/total*100:.1f}%)")

    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Extract verification results from MIP final.json files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to JSON (default)
  python extract_verification_results.py data/polaris_hard_40_final_n1.json

  # Extract to CSV
  python extract_verification_results.py data/polaris_hard_40_final_n1.json -f csv

  # Specify output file
  python extract_verification_results.py data/polaris_hard_40_final_n1.json -o results.json
        """
    )
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output file path (default: auto-generated)')
    parser.add_argument('-f', '--format', choices=['csv', 'json'], default='json',
                        help='Output format (default: json)')

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Extract data
    try:
        extract_verification_data(args.input_file, args.output, args.format)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()