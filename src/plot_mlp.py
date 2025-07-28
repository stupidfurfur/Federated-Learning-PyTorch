import matplotlib
import matplotlib.pyplot as plt

# Use 'Agg' backend if not displaying interactively (e.g., in non-GUI environments)
matplotlib.use('Agg')

# Read and convert the training loss data to floats
with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_83.33_90.27_loss.txt', 'r') as file:
    train_loss_1 = [float(line.strip()) for line in file]

with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_96.67_86.80_IDA_loss.txt', 'r') as file:
    train_loss_2 = [float(line.strip()) for line in file]

with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_95.00_84.75_ACC_INVERSE_loss.txt', 'r') as file:
    train_loss_3 = [float(line.strip()) for line in file]

with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_90.00_91.02_ACC_ACCUMULATE_loss.txt', 'r') as file:
    train_loss_4 = [float(line.strip()) for line in file]
# Plot Loss curve
plt.figure()
plt.title('Training Loss vs Communication Rounds')
plt.plot(range(len(train_loss_1)), train_loss_1, color='r', label='FedAvg')
plt.plot(range(len(train_loss_2)), train_loss_2, color='b', label='IDA')
plt.plot(range(len(train_loss_3)), train_loss_3, color='g', label='ACC inverse')
plt.plot(range(len(train_loss_4)), train_loss_4, color='purple', label='ABAVG')
# Labels and legend
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.legend()

plt.savefig('training_loss_plot_mlp_50_loss.png')

with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_83.33_90.27_acc.txt', 'r') as file:
    train_acc_1 = [float(line.strip()) for line in file]

with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_96.67_86.80_IDA_acc.txt', 'r') as file:
    train_acc_2 = [float(line.strip()) for line in file]
with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_95.00_84.75_ACC_INVERSE_acc.txt', 'r') as file:
    train_acc_3 = [float(line.strip()) for line in file]

with open('mnist_mlp_50_C[0.1]_iid[0]_E[10]_B[10]_90.00_91.02_ACC_ACCUMULATE_acc.txt', 'r') as file:
    train_acc_4 = [float(line.strip()) for line in file]
# Plot Loss curve
plt.figure()
plt.title('Average Accuracy vs Communication rounds')
plt.plot(range(len(train_acc_1)), train_acc_1, color='r', label='FedAvg')
plt.plot(range(len(train_acc_2)), train_acc_2, color='b', label='IDA')
plt.plot(range(len(train_acc_3)), train_acc_3, color='g', label='ACC inverse')
plt.plot(range(len(train_acc_4)), train_acc_4, color='purple', label='ABAVG')
# Labels and legend
plt.ylabel('Average Accuracy')
plt.xlabel('Communication Rounds')
plt.legend()

plt.savefig('training_loss_plot_mlp_50_acc.png')

categories = ['FedAvg', 'IDA', 'ACC inverse', 'ABAVG']
train_acc = [83.33,96.67,95.0,90.0]
test_acc = [90.27, 86.8, 84.75, 91.02]

# Define specific colors for each bar
colors = ['skyblue', 'deepskyblue', 'dodgerblue', 'cornflowerblue']  # Colors for test accuracy bars

# Create the figure with the specified size
plt.figure(figsize=(8, 4))  # Only call plt.figure once and pass figsize

# Create the bar chart for the first subplot (Avg Train Accuracy)
plt.subplot(1, 2, 1)
plt.bar(categories, train_acc, color=colors)  # Use the defined colors for train accuracy
plt.title('Avg Train Accuracy')
plt.xlabel('Methods')
plt.ylabel('Accuracy')

# Create the bar chart for the second subplot (Avg Test Accuracy)
plt.subplot(1, 2, 2)
plt.bar(categories, test_acc, color=colors)  # Use the defined colors for test accuracy
plt.title('Test Accuracy')
plt.xlabel('Methods')
plt.ylabel('Accuracy')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot to a file
plt.savefig('training_loss_plot_mlp_50_bar.png')