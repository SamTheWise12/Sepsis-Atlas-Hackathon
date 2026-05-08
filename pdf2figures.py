import subprocess
from pathlib import Path

if __name__=="__main__":
    # Paths
    base_dir = Path(__file__).parent.resolve()
    jar_path = base_dir / "pdffigures2.jar"
    pdf_path = base_dir / "Baloch_2022.pdf"
    output_dir = base_dir / "output"

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "java", "-jar", str(jar_path),
        str(pdf_path),
        "-g", str(output_dir / "json" / "metadata.json"),
        "-m", str(output_dir / "figures") + "/" 
    ], check=True)

    print("Done! Check the output folder.")