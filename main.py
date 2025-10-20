
import subprocess
import os

def main():
    """
    Runs the Streamlit app using a hardcoded, correct Python interpreter.
    """
    app_path = os.path.join(os.path.dirname(__file__), 'app', 'streamlit_app.py')

    # Hardcode the path to the correct python executable
    python_exe = "C:\\Python313\\python.exe"
    command = [python_exe, "-m", "streamlit", "run", app_path]

    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Error: 'streamlit' command not found.")
        print("Please make sure Streamlit is installed: pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the Streamlit app: {e}")

if __name__ == "__main__":
    main()
