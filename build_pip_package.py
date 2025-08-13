import os
import shutil
import subprocess
import sys
import tempfile

def is_nightly():
    return os.environ.get("IS_NIGHTLY") == "nightly"

def abspath(path):
    return os.path.abspath(path)

def main(output_dir=None):
    if output_dir is None:
        output_dir = "/tmp/tensorflow_text_pkg"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = abspath(output_dir)

    print(f"=== Destination directory: {output_dir}")
    print(f"=== Current directory: {os.getcwd()}")

    # Source directories/files in your repo (relative to repo root)
    src_tensorflow_text = os.path.join("tensorflow_text")
    src_setup_py = os.path.join("oss_scripts", "pip_package", "setup.nightly.py" if is_nightly() else "setup.py")
    src_manifest = os.path.join("oss_scripts", "pip_package", "MANIFEST.in")
    src_license = os.path.join("oss_scripts", "pip_package", "LICENSE")

    # Check if source exists
    if not os.path.isdir(src_tensorflow_text):
        print(f"Error: directory '{src_tensorflow_text}' not found.", file=sys.stderr)
        sys.exit(1)
    for f in [src_setup_py, src_manifest, src_license]:
        if not os.path.isfile(f):
            print(f"Error: file '{f}' not found.", file=sys.stderr)
            sys.exit(1)

    temp_dir = tempfile.mkdtemp()
    print(f"=== Using tmpdir {temp_dir}")

    # Copy tensorflow_text directory
    dest_tf_text = os.path.join(temp_dir, "tensorflow_text")
    shutil.copytree(src_tensorflow_text, dest_tf_text)

    # Copy setup.py or setup.nightly.py
    shutil.copy(src_setup_py, temp_dir)
    setup_script = os.path.basename(src_setup_py)

    # Copy MANIFEST.in and LICENSE
    shutil.copy(src_manifest, temp_dir)
    shutil.copy(src_license, temp_dir)

    # Run setup.py bdist_wheel --universal
    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        python_cmd = shutil.which("python3") or shutil.which("python")
        if python_cmd is None:
            print("Python not found in PATH.", file=sys.stderr)
            sys.exit(1)

        subprocess.run([python_cmd, setup_script, "bdist_wheel", "--universal"], check=True)
        # Copy generated wheel(s) to output directory
        dist_dir = os.path.join(temp_dir, "dist")
        for filename in os.listdir(dist_dir):
            if filename.endswith(".whl"):
                shutil.copy(os.path.join(dist_dir, filename), output_dir)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    output_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_dir_arg)
