#!/bin/bash
echo "Setting up Health Hub virtual environment and installing dependencies..."

# Define virtual environment directory
VENV_DIR="venv_health_hub" # More specific name for clarity

# Check if Python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 command could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ./${VENV_DIR}..."
    python3 -m venv ${VENV_DIR}
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment ./${VENV_DIR} already exists."
fi


# Activate virtual environment
# Source command might differ slightly based on shell (bash/zsh vs fish)
echo "Activating virtual environment..."
source ./${VENV_DIR}/bin/activate

# Check if activation was successful (relies on VIRTUAL_ENV being set)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment activation failed."
    echo "Please try activating manually: source ./${VENV_DIR}/bin/activate"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip within the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements from requirements.txt
REQUIREMENTS_FILE="requirements.txt" # Assumed to be in the same directory as setup.sh
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
    pip install -r ${REQUIREMENTS_FILE}
    if [ $? -eq 0 ]; then
        echo "Dependencies installed successfully."
    else
        echo "Error: Failed to install dependencies from ${REQUIREMENTS_FILE}."
        echo "Please check the file for errors and try again."
        exit 1
    fi
else
    echo "Warning: ${REQUIREMENTS_FILE} not found. Skipping dependency installation."
    echo "Please ensure you have a requirements.txt file with necessary packages (e.g., streamlit, pandas, geopandas, plotly, numpy)."
fi

echo ""
echo "Setup complete."
echo "To run the application:"
echo "1. Ensure the virtual environment is active: source ./${VENV_DIR}/bin/activate"
echo "2. Navigate to the 'health_hub' directory (if not already there)."
echo "3. Run: streamlit run app_home.py"
echo ""
echo "To deactivate the virtual environment later, simply type: deactivate"
