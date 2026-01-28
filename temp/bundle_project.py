import os
import json
from pathlib import Path

def generate_directory_structure(root_dir, skip_dirs=None, skip_files=None):
    if skip_dirs is None:
        skip_dirs = {'.git', 'docs', 'temp', '__pycache__'}
    if skip_files is None:
        skip_files = {'.gitignore', 'README.md', 'requirements.txt', 'pyproject.toml'}
    
    structure = []
    for root, dirs, files in os.walk(root_dir):
        # Modifier dirs in-place to skip
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if f not in skip_files:
                structure.append(f"{sub_indent}{f}")
                
    return "\n".join(structure)

def bundle_project(root_path, output_json):
    root = Path(root_path).resolve()
    skip_dirs = {'.git', 'docs', 'temp', '__pycache__'}
    skip_files = {'.gitignore', 'README.md', 'requirements.txt', 'pyproject.toml'}
    
    bundle = {
        "introduction": (
            "PyDetonation-Ultra (PDU) V8.2 Source Code Bundle\n"
            "This bundle contains the core physical engine, data, and verification scripts "
            "for the PDU V8.2 milestone, excluding documentation and project meta-files."
        ),
        "directory_structure": generate_directory_structure(str(root), skip_dirs, skip_files),
        "files": {}
    }
    
    for root_dir, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file in skip_files:
                continue
                
            file_path = Path(root_dir) / file
            relative_path = file_path.relative_to(root)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                bundle["files"][str(relative_path)] = content
            except Exception as e:
                bundle["files"][str(relative_path)] = f"Error reading file: {str(e)}"
                
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdu_project_bundle.json")
    print(f"Bundling project from {project_root} to {output_file}...")
    bundle_project(project_root, output_file)
    print("Done.")
