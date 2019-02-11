import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import probplot, pearsonr


class PreparedData:
    def __init__(self, inn):
        self.original_data = inn
        self.prepared_data = None
        self.feature_labels = None
        self.target_labels = None
        # Assign index and convert to date.
        out = inn.copy(deep=True)
        out.index = pd.to_datetime(inn['date'])
        # Drop unnecessary columns.
        out.drop(columns=['date', 'rv1', 'rv2'], inplace=True)
        # Get labels.
        target_labels = [
            label for label in inn.columns if
            ('T' in label)
            & ('_out' not in label)
            & ('Tdewpoint' not in label)
        ]
        feature_labels = set(out.columns) - set(target_labels)
        self.prepared_data = out
        self.feature_labels = feature_labels
        self.target_labels = target_labels


def compute_univariate_statistics(inn):
    """
    Return a data frame wh. univariate statistics.
    """
    # Compute univariate statistics.
    statistics = inn.describe().T
    # Compute skewness.
    statistics['skewness'] = inn.skew().values
    # Compute kurtosis.
    statistics['kurtosis'] = inn.kurtosis().values
    # Compute inter-quartile range.
    statistics['IQR'] = statistics['75%'] - statistics['25%']
    # Declare a list to receive a
    proportion_of_outliers = []
    # Compute proportion of outilers within each variable according to the
    # Tukey's fences.
    for variable in enumerate(inn.columns):
        proportion_of_outliers.append(inn[variable[1]][(
            inn[variable[1]] <
                statistics['25%'][variable[0]]
                - (1.5*statistics['25%'][variable[0]])
            ) | (
            inn[variable[1]] >
                statistics['75%'][variable[0]]
                + (1.5*statistics['75%'][variable[0]])
        )].count()/len(inn))
    statistics['proportion_of_outliers'] = np.around(
        np.array(proportion_of_outliers), decimals=2
    )
    # Drop IQR.
    statistics.drop('IQR', axis=1, inplace=True)
    return statistics


if __name__ == '__main__':

    # Declare file handles.
    RAW_DATA = \
        r'M:\Projects\003_univariate_and_multivariate_data_analysis\1_data' \
        + r'\energydata_complete.csv'

    # Get data.
    raw_data = pd.read_csv(RAW_DATA, sep=',')

    # Prepare data.
    data_preparation = PreparedData(inn=raw_data)
    prepared_data = data_preparation.prepared_data

    # Compute univariate statistics.
    univariate_statistics = compute_univariate_statistics(inn=prepared_data)

    # Plot time-series.
    # Declarea a figure.
    fig = plt.figure()
    grid = gridspec.GridSpec(2, 3)
    # Plot time series.
    ax1 = plt.subplot(grid[0, :3])
    ax1.plot(prepared_data['T1'])
    # Plot normal probability plot.
    ordered_series = np.array(prepared_data['T1'].sort_values(ascending=True))
    normal_distribution = np.sort(
        np.random.normal(loc=0, scale=1, size=len(ordered_series))
    )
    ax2 = plt.subplot(grid[1, 0])
    ax2.scatter(x=normal_distribution, y=ordered_series, s=1)
    ax2.plot(
        [np.min(normal_distribution), np.max(normal_distribution)],
        [np.min(ordered_series), np.max(ordered_series)],
        c='red'
    )
    ax2.set_xlabel('Theoretical Distribution')
    ax2.set_ylabel('Observed Distribution')
    ax2.set_title('Normal Probability Plot')
    # Plot histogram.
    ax3 = plt.subplot(grid[1, 1])
    ax3.hist(prepared_data['T1'], bins=30)
    ax3.set_title('Histogram')
    # Plot univariate statisticsl.
    ax3 = plt.subplot(grid[1, 2])
    row_labels = list(univariate_statistics.columns)
    content = [[item] for item in list(np.around(univariate_statistics.loc['T1', :].values, decimals=4))]
    ax3.table(
        cellText=content,
        rowLabels=row_labels,
        loc='right',
        colWidths=[0.5, 0.5], bbox=None, fontsize=10).scale(.8, .8)
    ax3.axis('off')
    ax3.axis('tight')
    ax3.set_title('Univarite Statistics')
