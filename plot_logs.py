import matplotlib.pyplot as plt
import pandas as pd


def plot_grb_res(grb_csv_file, ax=None):
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
    # ax.figure(figsize=(10, 5))
    if ax:
        ax.plot(df['Time'], df['Incumbent'], label='Incumbent')
        ax.plot(df['Time'], df['BestBd'], label='BestBd')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Objective Value')
    else:
        plt.plot(df['Time'], df['Incumbent'], label='Incumbent')
        plt.plot(df['Time'], df['BestBd'], label='BestBd')
        plt.xlabel('Time(s)')
        plt.ylabel('Objective Value')
        # plt.title('Incumbent and BestBd vs Time')
        # plt.legend()
        # plt.savefig('Incumbent_and_BestBd_vs_Time.png')
        

def plot_bcd_res(bcd_csv_files: list, ax=None):
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
        ax = ax if ax else plt
        ax.plot(df['t'].iloc[-1], df["c'x"].iloc[-1], label=label, marker='o', markersize=3)
        # plt.xlabel('t')
        # plt.ylabel("c'x")
        # plt.title(f"c'x vs t for {file}")
        # plt.legend()
        # plt.savefig(f"c'x_vs_t_for_{file}.png")
        # plt.show()
        
if __name__ == "__main__":
    # Load the data
    p_data = pd.read_csv('C109_p_res50.csv')
    c_data = pd.read_csv('C109_c_res50.csv')

    # Define a larger font size
    font_size = 28
    # set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': font_size})

    # Plot the first graph
    plt.figure(figsize=(10, 6))
    plt.plot(p_data["t"], p_data["c'x"], label='alm-p')
    plt.plot(c_data["t"], c_data["c'x"], label='alm-c')
    plt.xlabel('Time(s)')
    plt.ylabel("c'x")
    plt.legend()
    # plt.title('Objective Value')
    plt.tight_layout()
    plt.savefig('objective_value.png')
    plt.close()

    # Plot the second graph
    plt.figure(figsize=(10, 6))
    plt.plot(p_data["t"], p_data["|Ax - b|"], label='alm-p')
    plt.plot(c_data["t"], c_data["|Ax - b|"], label='alm-c')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time(s)')
    plt.ylabel("|Ax - b|")
    plt.legend()
    plt.tight_layout()
    # plt.title('Constraint Violation')
    plt.savefig('constraint_violation.png')
    plt.close()

    # Plot the third graph
    plt.figure(figsize=(10, 6))
    plot_grb_res('c109.50.clean.csv')
    plot_bcd_res(['C109_p_res50.csv', 'C109_c_res50.csv'])
    plt.legend()
    # plt.title('Primal and Dual Bound of Gurobi')
    plt.tight_layout()
    plt.savefig('primal_dual_bound.png')
    plt.close()