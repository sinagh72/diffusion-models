import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.ma.extras import average


def plot_task_1(df_1):

    label_counts = df_1.groupby('label').size()
    # Plotting the bar chart
    colors = ['green', 'gray', 'red', 'blue']
    label_words = {
        '0': 'Reduced',
        '1': 'Stable',
        '2': 'Increased',
        '3': 'Uninterpretable'
    }

    # Plotting the bar chart
    plt.figure(figsize=(8, 6))
    bars = label_counts.plot(kind='bar', color=colors)

    # Assigning labels to each bar
    plt.xlabel('Category')
    plt.ylabel('Number of OCT slices')
    plt.title('Number of OCT slices per category')

    # Changing the x-axis labels to the corresponding words
    plt.xticks(ticks=range(len(label_counts)), labels=[label_words[str(label)] for label in label_counts.index], rotation=0)
    # Adding the total number on top of each bar
    for i, value in enumerate(label_counts):
        plt.text(i, value + 10, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Manually creating legend handles and labels
    handles = [mpatches.Patch(color=colors[i], label=label_words[str(label)]) for i, label in
               enumerate(label_counts.index)]
    plt.legend(handles=handles, title="Categories")
    # Adding a legend (caption) for the colors with associated words

    # Displaying the bar chart
    plt.show()


def plot_task_2(df_2):

    label_counts = df_2.groupby('label').size()
    print(label_counts)
    # Plotting the bar chart
    colors = ['green', 'gray', 'red']
    label_words = {
        '0.0': 'Reduced',
        '1.0': 'Stable',
        '2.0': 'Increased',
    }

    # Plotting the bar chart
    plt.figure(figsize=(8, 6))
    bars = label_counts.plot(kind='bar', color=colors)

    # Assigning labels to each bar
    plt.xlabel('Category')
    plt.ylabel('Number of OCT slices')
    plt.title('Number of OCT slices per category')

    # Changing the x-axis labels to the corresponding words
    plt.xticks(ticks=range(len(label_counts)), labels=[label_words[str(label)] for label in label_counts.index], rotation=0)
    # Adding the total number on top of each bar
    for i, value in enumerate(label_counts):
        plt.text(i, value + 10, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Manually creating legend handles and labels
    handles = [mpatches.Patch(color=colors[i], label=label_words[str(label)]) for i, label in
               enumerate(label_counts.index)]
    plt.legend(handles=handles, title="Categories")
    # Adding a legend (caption) for the colors with associated words

    # Displaying the bar chart
    plt.show()

if __name__ == "__main__":
    root = "/data1/OCT/Mario/data_2"
    df_2 = pd.read_csv(os.path.join(root, "df_task2_train_challenge.csv"))
    patient_counts = df_2.groupby('id_patient').size()
    print(patient_counts.max())
    print(patient_counts.min())
    print(patient_counts.mean())
    plot_task_2(df_2)
    # print(min(patient_counts))
    # print(average(patient_counts))
