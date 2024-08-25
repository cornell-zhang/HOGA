import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_absolute_percentage_error
import statistics
import argparse

# Function to extract trainMode from file name
def get_train_mode(file_name):
    parts = file_name.split('_')
    if len(parts) == 3:
        return parts[0], parts[1], parts[2].replace('.csv', '')
    elif len(parts) == 4:
        name = parts[2] + '_' + parts[3].replace('.csv', '')
        return parts[0], parts[1], name
    else:
        return "unknown", "unknown", "unknown"

def main(args):
    # Set the path where the CSV files are located
    path = args.input_path
    pattern = re.compile(r"^[a-zA-Z0-9]+_[a-zA-Z]+_[a-zA-Z_0-9]+\.csv$")


    # Set the style for seaborn
    sns.set_style("white")

    # Process each CSV file in the directory
    mape_list = []
    for file in os.listdir(path):
        print(file)
        if pattern.match(file):
            # Extract trainMode from the file name
            DS, trainMode,name = get_train_mode(file)
            if trainMode != 'test' or DS != 'desDF1':
                continue

            # Read the CSV file
            df = pd.read_csv(os.path.join(path, file))

            print(df.columns)  # To check all column names in the DataFrame
            print(df.head())   # To print the first few rows of the DataFrame
            x = df['prediction']
            y = df['actual']
            mape_score = mean_absolute_percentage_error(y.to_list(),x.to_list())
            mape_list.append(mape_score)
            print(f"{name} MAPE: ", mape_score)

            print("\nDataset type: " + trainMode)

            # Create a scatter plot using Seaborn
            plt.figure(figsize=(8, 6))  # Adjust size as needed
            ax = sns.scatterplot(x='actual', y='prediction', data=df, color='#334488', s=50, alpha=0.5, edgecolor='none')
            sns.regplot(x="actual", y="prediction", data=df, scatter=False, color='red')
            plt.title(f"{name} (HOGA)", fontsize=25, pad=10)
            plt.xlabel('Ground Truth', fontsize=20)
            plt.ylabel('Prediction', fontsize=20, labelpad=-5)
            ax.set_xlim([-2, 6])  # Adjust these values based on your data and preferences
            ax.set_ylim([-2, 6])
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Save the plot
            dumpDir = args.output_path
            fileName = os.path.join(dumpDir, f"hoga_{name}.pdf")
            plt.savefig(fileName, format='pdf', bbox_inches='tight')
            plt.close()
    mape_avg = statistics.mean(mape_list)
    print("Average MAPE Score: ", mape_avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./hoga_logs", help="path to the input directory")
    parser.add_argument("--output_path", type=str, default="./figures", help="path to the output directory")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    main(args)
