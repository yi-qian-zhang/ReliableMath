#!/usr/bin/env python3
"""
Diagnostic test to verify API is working correctly
Run this before the main pipeline to ensure your setup is correct
"""

import json
from openai import OpenAI

# Load API configuration
try:
    model_options = json.load(open("./data/api_keys.json", "r"))
except FileNotFoundError:
    print("ERROR: api_keys.json not found in ./data/")
    print("Please create it with your API credentials")
    exit(1)

def test_single_extraction():
    """Test extraction on a single problem"""
    
    test_problem = "The front tires of a car wear out after 25,000 km, and the rear tires wear out after 15,000 km. When should the tires be swapped so that they wear out at the same time?"
    
    prompt = f"""Extract 1-3 key mathematical conditions from the following problem.

Problem: {test_problem}

Instructions:
- Extract only the essential conditions that, if removed, would make the problem unsolvable
- For word problems: extract numerical values, constraints, or goals
- Output each condition on a separate line
- Do not include numbering or bullet points

Extracted conditions:"""
    
    print("="*70)
    print("API CONFIGURATION TEST")
    print("="*70)
    
    # Test each configured model
    for model_key in model_options.keys():
        print(f"\nTesting model: {model_key}")
        print("-"*40)
        
        try:
            model_name, key, url = model_options[model_key][0]
            client = OpenAI(api_key=key, base_url=url)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            print(f"✓ API call successful")
            print(f"Response preview: {result[:200]}...")
            
            # Check if response is relevant
            result_lower = result.lower()
            if any(word in result_lower for word in ['tire', 'wheel', '25', '15', 'km', 'front', 'rear']):
                print("✓ Response is RELEVANT - contains tire/wheel information")
            else:
                print("✗ WARNING: Response may be IRRELEVANT")
                print(f"Full response:\n{result}\n")
                print("If you see content about rectangles or other unrelated topics,")
                print("there may be an issue with your API configuration.")
            
        except Exception as e:
            print(f"✗ Error testing {model_key}: {e}")
    
    print("\n" + "="*70)

def test_all_problems():
    """Test on multiple problem types"""
    
    test_problems = [
        {
            "type": "Tire problem",
            "question": "The front tires of a car wear out after 25,000 km, and the rear tires wear out after 15,000 km. When should the tires be swapped so that they wear out at the same time?",
            "expected_keywords": ['tire', 'wheel', '25', '15', 'km', 'front', 'rear', 'wear', 'swap']
        },
        {
            "type": "Tetrahedron problem", 
            "question": "The edges of tetrahedron $ABCD$ are given as $AB=c$, $BC=a$, $CA=b$, $DA=a_{1}$, $DB=b_{1}$, and $DC=c_{1}$. Let $h$ be the length of the median line from vertex $D$.",
            "expected_keywords": ['edge', 'tetrahedron', 'vertex', 'median', 'length', 'ab', 'bc', 'ca']
        }
    ]
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST ON MULTIPLE PROBLEMS")
    print("="*70)
    
    # Use the first available model
    model_key = list(model_options.keys())[0]
    model_name, key, url = model_options[model_key][0]
    client = OpenAI(api_key=key, base_url=url)
    
    for problem_data in test_problems:
        print(f"\n{problem_data['type']}")
        print("-"*40)
        print(f"Question: {problem_data['question'][:100]}...")
        
        prompt = f"""Extract key conditions from this problem:
{problem_data['question']}

Output only the conditions, one per line:"""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            result_lower = result.lower()
            
            # Check for expected keywords
            found_keywords = [kw for kw in problem_data['expected_keywords'] if kw in result_lower]
            
            print(f"Response: {result[:150]}...")
            
            if len(found_keywords) >= 2:
                print(f"✓ PASS - Found keywords: {found_keywords[:5]}")
            else:
                print(f"✗ FAIL - Missing expected content")
                print(f"Expected keywords: {problem_data['expected_keywords']}")
                print(f"Found keywords: {found_keywords}")
                print(f"Full response:\n{result}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "="*70)

def validate_api_keys():
    """Validate the structure of api_keys.json"""
    
    print("\n" + "="*70)
    print("API KEYS VALIDATION")
    print("="*70)
    
    for model_key, configs in model_options.items():
        print(f"\nModel: {model_key}")
        
        if not isinstance(configs, list):
            print(f"✗ Error: Configuration should be a list")
            continue
        
        for i, config in enumerate(configs):
            if not isinstance(config, list) or len(config) != 3:
                print(f"✗ Config {i}: Should be [model_name, api_key, base_url]")
                continue
            
            model_name, api_key, base_url = config
            
            # Basic validation
            checks = []
            
            if model_name and isinstance(model_name, str):
                checks.append("✓ Model name OK")
            else:
                checks.append("✗ Model name missing or invalid")
            
            if api_key and isinstance(api_key, str) and len(api_key) > 10:
                checks.append("✓ API key present")
            else:
                checks.append("✗ API key missing or too short")
            
            if base_url and isinstance(base_url, str) and base_url.startswith("http"):
                checks.append("✓ Base URL OK")
            else:
                checks.append("✗ Base URL invalid (should start with http)")
            
            print(f"  Config {i}: {', '.join(checks)}")
    
    print("="*70)

if __name__ == "__main__":
    print("\nStarting diagnostic tests...\n")
    
    # Validate configuration structure
    validate_api_keys()
    
    # Test single extraction
    test_single_extraction()
    
    # Test multiple problem types
    test_all_problems()
    
    print("\nDiagnostic tests completed!")
    print("\nIf all tests passed, you can run the main pipeline:")
    print("  python main.py --dataset polaris --test_mode")
    print("\nIf tests failed, please check:")
    print("1. Your API credentials in ./data/api_keys.json")
    print("2. The model name (e.g., 'deepseek-chat' not 'deepseek_v3')")
    print("3. The API endpoint URL")