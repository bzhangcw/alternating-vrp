import matplotlib.pyplot as plt
import pandas as pd


def plot_grb_res(grb_csv_file):
    df = pd.read_csv(grb_csv_file)

    # Drop duplicates, keeping only the last occurrence
    df = df.drop_duplicates(subset='Time', keep='last')

    # Convert 'Time' column to numeric
    df['Time'] = pd.to_numeric(df['Time'])

    # Convert 'Incumbent' and 'BestBd' columns to numeric, errors='coerce' will replace non-numeric values with NaN
    # and drop Nan values
    df['Incumbent'] = pd.to_numeric(df['Incumbent'], errors='coerce')
    df['BestBd'] = pd.to_numeric(df['BestBd'], errors='coerce')

    # Plot Incumbent vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Incumbent'], label='Incumbent')
    plt.plot(df['Time'], df['BestBd'], label='BestBd')
    plt.xlabel('Time')
    plt.ylabel('Objective Value')
    # plt.title('Incumbent and BestBd vs Time')
    # plt.legend()
    # plt.savefig('Incumbent_and_BestBd_vs_Time.png')
    # plt.show()

def plot_bcd_res(bcd_csv_files: list):
    for file in bcd_csv_files:
        df = pd.read_csv(file)

        # Convert 't' column to numeric
        df['t'] = pd.to_numeric(df['t'])

        # Convert 'c'x' and 'c'x (H)' columns to numeric, errors='coerce' will replace non-numeric values with NaN
        # and drop Nan values
        df["c'x"] = pd.to_numeric(df["c'x"], errors='coerce')
        df["c'x (H)"] = pd.to_numeric(df["c'x (H)"], errors='coerce')

        # Plot c'x vs t
        label = "alm-c" if "_c_" in file else "alm-p"
        plt.plot(df['t'].iloc[-1], df["c'x"].iloc[-1], label=label, marker='o', markersize=3)
        # plt.xlabel('t')
        # plt.ylabel("c'x")
        # plt.title(f"c'x vs t for {file}")
        # plt.legend()
        # plt.savefig(f"c'x_vs_t_for_{file}.png")
        # plt.show()
        
if __name__ == "__main__":
    # plot all together in a single plot
    problem_name = "r105.50"
    plot_grb_res('r105.50.clean.csv')
    plot_bcd_res(['r105_c_res50.csv', 'r105_p_res50.csv'])
    plt.legend()
    plt.savefig(f'{problem_name}_res.png')
