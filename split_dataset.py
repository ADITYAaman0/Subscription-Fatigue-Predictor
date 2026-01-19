import os
import shutil
from pathlib import Path

# Configuration
input_zip = r'x:\spc\data\raw\ecommerce-behavior-data-from-multi-category-store\ecommerce-behavior-data-from-multi-category-store.zip'
output_dir = r'x:\spc\data\raw\ecommerce-behavior-data-from-multi-category-store\split_chunks'
chunk_size_mb = 24
chunk_size_bytes = chunk_size_mb * 1024 * 1024

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Get file info
file_size = os.path.getsize(input_zip)
total_chunks = (file_size + chunk_size_bytes - 1) // chunk_size_bytes

print(f"Total file size: {file_size / (1024**3):.2f} GB ({file_size / (1024**2):.2f} MB)")
print(f"Chunk size: {chunk_size_mb} MB")
print(f"Total chunks needed: {total_chunks}")
print(f"Output directory: {output_dir}\n")

# Split the file
with open(input_zip, 'rb') as input_file:
    for chunk_num in range(total_chunks):
        chunk_filename = os.path.join(output_dir, f'ecommerce_chunk_{chunk_num + 1:03d}.zip')
        bytes_read = 0
        
        with open(chunk_filename, 'wb') as chunk_file:
            while bytes_read < chunk_size_bytes:
                remaining = chunk_size_bytes - bytes_read
                data = input_file.read(min(remaining, 8192))
                if not data:
                    break
                chunk_file.write(data)
                bytes_read += len(data)
        
        chunk_file_size = os.path.getsize(chunk_filename)
        print(f"Created: {os.path.basename(chunk_filename)} ({chunk_file_size / (1024**2):.2f} MB)")

print(f"\nSplitting complete! All chunks are in: {output_dir}")

# Create a reassembly script
reassembly_script = os.path.join(output_dir, 'REASSEMBLE_README.txt')
with open(reassembly_script, 'w') as f:
    f.write("INSTRUCTIONS TO REASSEMBLE THE DATASET\n")
    f.write("=" * 50 + "\n\n")
    f.write("This dataset has been split into chunks of 24 MB for GitHub upload.\n\n")
    f.write("To reassemble:\n")
    f.write("1. Download all chunks (ecommerce_chunk_001.zip through ecommerce_chunk_*.zip)\n")
    f.write("2. Place them in the same directory\n")
    f.write("3. Run the following command in PowerShell:\n\n")
    f.write("   $output = 'ecommerce-behavior-data-from-multi-category-store.zip'\n")
    f.write("   $chunks = Get-ChildItem ecommerce_chunk_*.zip | Sort-Object Name\n")
    f.write("   $out = [System.IO.File]::Create($output)\n")
    f.write("   foreach ($chunk in $chunks) {\n")
    f.write("       $bytes = [System.IO.File]::ReadAllBytes($chunk.FullName)\n")
    f.write("       $out.Write($bytes, 0, $bytes.Length)\n")
    f.write("       Write-Host \"Appended $($chunk.Name)\"\n")
    f.write("   }\n")
    f.write("   $out.Close()\n")
    f.write("   Write-Host 'Reassembly complete!'\n\n")
    f.write("4. Extract the reassembled zip file\n")

print(f"Reassembly instructions written to: {reassembly_script}")
