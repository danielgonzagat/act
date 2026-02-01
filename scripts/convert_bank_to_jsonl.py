
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert.py <input.json> <output.jsonl>")
        sys.exit(1)
        
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    
    data = json.loads(in_path.read_text())
    
    # Handle "schemas" key if present, otherwise assume list
    schemas = []
    if isinstance(data, dict) and "schemas" in data:
        schemas = data["schemas"]
    elif isinstance(data, list):
        schemas = data
    else:
        print("Error: Input must be a list or a dict with 'schemas' key")
        sys.exit(1)
        
    with open(out_path, "w") as f:
        for s in schemas:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Converted {len(schemas)} schemas to {out_path}")

if __name__ == "__main__":
    main()
