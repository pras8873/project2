
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "openai",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai

# Function to read the CSV file
def load_csv(filename):
    try:
        data = pd.read_csv(filename)
        print(f"Loaded {filename} successfully.")
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

# Function to perform basic data inspection
def inspect_data(data):
    report = {}
    report['columns'] = data.columns.tolist()
    report['dtypes'] = data.dtypes.astype(str).to_dict()
    report['missing_values'] = data.isnull().sum().to_dict()
    report['summary_statistics'] = data.describe(include='all').to_dict()
    return report

# Function to generate visualizations
def create_visualizations(data, output_prefix):
    # Visualization 1: Correlation heatmap for numerical data
    numerical_data = data.select_dtypes(include=['number'])
    if not numerical_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.savefig(f"{output_prefix}_correlation_heatmap.png")
        plt.close()
        print(f"Saved {output_prefix}_correlation_heatmap.png")

    # Visualization 2: Missing value heatmap
    if data.isnull().any().any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.savefig(f"{output_prefix}_missing_values.png")
        plt.close()
        print(f"Saved {output_prefix}_missing_values.png")

    # Visualization 3: Example boxplot for the first numerical column (if any)
    if not numerical_data.empty:
        first_col = numerical_data.columns[0]
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[first_col])
        plt.title(f"Boxplot of {first_col}")
        plt.savefig(f"{output_prefix}_boxplot.png")
        plt.close()
        print(f"Saved {output_prefix}_boxplot.png")

# Function to generate narrative using LLM
def generate_narrative(data_report, output_prefix):
    token = os.environ.get("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjIwMDA5NzNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.gazqHNeKzelLyQYETDKkuGaj4e4DAhYFhtgfXM8n3rk")
    if not token:
        print("AIPROXY_TOKEN environment variable is not set. Exiting.")
        sys.exit(1)
    
    openai.api_key = token
    prompt = f"""
    Analyze the following dataset summary and generate a story:
    Columns and Types: {data_report['dtypes']}
    Missing Values: {data_report['missing_values']}
    Summary Statistics: {data_report['summary_statistics']}
    Provide a story describing the dataset, key insights, and actionable conclusions.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        narrative = response['choices'][0]['message']['content']
        with open(f"{output_prefix}_README.md", "w") as f:
            f.write(narrative)
        print(f"Saved narrative as {output_prefix}_README.md")
    except Exception as e:
        print(f"Error generating narrative: {e}")

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    output_prefix = os.path.splitext(filename)[0]
    data = load_csv(filename)
    report = inspect_data(data)
    create_visualizations(data, output_prefix)
    generate_narrative(report, output_prefix)

if __name__ == "__main__":
    main()
