import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_series(data: pd.Series, title: str = '', figsize: tuple = (12, 6)) -> None:
    # Create figure and axes
    plt.figure(figsize=figsize)
    
    # Create line plot
    sns.lineplot(
        data=data,        
    )    
    # Customize plot
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to prevent label cutoff

def plot_series2(data1: pd.Series, data2: pd.Series, title: str = '', figsize: tuple = (12, 6), labels: tuple = ('Series 1', 'Series 2')) -> None:
    # Create figure and axes
    plt.figure(figsize=figsize)
    
    # Create line plots
    sns.lineplot(data=data1, label=labels[0])    
    sns.lineplot(data=data2, label=labels[1])
    
    # Customize plot
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()  # Add legend to distinguish the lines
    plt.tight_layout()  # Adjust layout to prevent label cutoff

