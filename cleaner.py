import os
import re

def clean_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove specific unwanted parts
    patterns_to_remove = [
        r'بطاقة التشريع النوع:.*?الصفحة  من: \d+\s*',
        r'طباعة\s*',
        r'الرجاء عدم اعتبار المادة المعروضة أعلاه رسمية\s*',
        r'© 2017 حكومة دولة قطر\. حميع الحقوق محفوظة\.\s*'
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Remove extra newlines and spaces
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r' {2,}', ' ', content)
    content = content.strip()

    # Split content into lines
    lines = content.split('\n')

    # Remove exact duplicates while preserving order
    cleaned_lines = []
    seen_lines = set()
    for line in lines:
        if line.strip() and line not in seen_lines:
            cleaned_lines.append(line)
            seen_lines.add(line)

    # Combine cleaned content
    cleaned_content = '\n\n'.join(cleaned_lines)

    if cleaned_content.strip():
        # Write the cleaned content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        print(f"Cleaned: {file_path}")
    else:
        # Delete the file if it's empty after cleaning
        os.remove(file_path)
        print(f"Deleted empty file: {file_path}")

def scan_and_clean_folders():
    # List of folders created by the previous web scraping script
    folders = ['LawView', 'RulingView', 'ViewAgreement', 'OpinionView']
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"Processing folder: {folder}")
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        clean_file(file_path)
        else:
            print(f"Folder not found: {folder}")

if __name__ == "__main__":
    print("Starting the cleaning process...")
    scan_and_clean_folders()
    print("Cleaning process completed.")