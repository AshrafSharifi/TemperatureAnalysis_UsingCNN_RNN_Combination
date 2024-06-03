import subprocess

def run_script(script_name):
    try:
        # Call the script and wait for it to complete
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        # Print the output from the script
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # Print the error if the script fails
        print(f"Error running {script_name}: {e.stderr}")

# List of scripts to run in order
scripts = ['RNN&CNN.py', 'CNN_followedby_LSTM.py', 'LSTM_followedby_CNN.py']

# Run each script in order
for script in scripts:
    run_script(script)
