import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


# Creates a confusion matrix given labels(y_true),predictions(y_pred) and labels for confusion matrix(class_labels)
# Y_true and Y_pred must be ints, using metic_calc will help with this task
def create_confusion_matrix(y_true, y_pred, graph_name):
    class_labels = ["0","1","7","8","9","k","l","m","n","p"]

    cfm = tf.math.confusion_matrix(y_true, y_pred)

    sns.heatmap(cfm, annot=True, cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
    #plt.figure() #create new plot
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(graph_name)
    plt.close() #saves plot


# creates accuracy vs Validation accuracy with epochs graph using model history

def create_acc_val_epoch_graph(history, graph_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.figure() #creates new figure
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)

    plt.savefig(graph_name)
    plt.close() #closes plot
