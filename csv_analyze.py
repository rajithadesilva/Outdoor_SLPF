import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PARTICLE_COUNT = 100
PARTICLE_STD = 2.0

CSV_PATH = f'amcl_output/csv/likelihood_stats_{PARTICLE_COUNT}_{PARTICLE_STD}.csv'

def visualize_likelihood_csv(csv_path, output_dir='amcl_output/csv'):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Filter out outliers where mean_gps_dist > 5
    df = df[df['mean_gps_dist'] <= 5]

    x_axis = 'mean_gps_dist'
    log_terms = ['mean_log_match', 'mean_log_structure', 'mean_log_penalty', 'mean_log_gps']
    df['mean_log_total'] = df[log_terms].sum(axis=1)

    for term in log_terms + ['mean_log_total']:
        plt.figure(figsize=(8,5))
        sns.regplot(x=x_axis, y=term, data=df, scatter_kws={'s':40}, line_kws={'color':'red'})
        plt.xlabel('Mean GPS Distance (filtered ≤ 5)')
        plt.ylabel(term)
        plt.title(f'{term} vs Mean GPS Distance (filtered ≤ 5) with Regression')
        plt.grid(True)
        filename = f'{term}_vs_gpsdist_filtered_regplot.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Regression plots (filtered) saved in folder: {output_dir}")

if __name__ == "__main__":
    visualize_likelihood_csv(CSV_PATH)

