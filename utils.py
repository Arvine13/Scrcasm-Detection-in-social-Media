import sys
import traceback

import seaborn as sbn
from PyQt5.QtCore import QAbstractTableModel, Qt, QObject, pyqtSignal, QRunnable, pyqtSlot
from PyQt5.QtWidgets import QMessageBox
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'JetBrains Mono'

CLASSES = ['Non-Sarcastic', 'Sarcastic']


class PandasDfToPyqtTable(QAbstractTableModel):
    def __init__(self, df):
        QAbstractTableModel.__init__(self)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=None):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return list(self._df.columns)[col]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return col
        return None


def clear_layout(layout):
    while layout.count() > 0:
        item = layout.takeAt(0)
        if not item:
            continue
        w = item.widget()
        if w:
            w.deleteLater()


def show_message_box(title, icon, msg):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(msg)
    msg_box.setIcon(icon)
    msg_box.setDefaultButton(QMessageBox.Ok)
    msg_box.setWindowModality(Qt.ApplicationModal)
    msg_box.exec_()


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    log_updater = pyqtSignal(object)
    table_adder = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.kwargs['log'] = self.signals.log_updater
        self.kwargs['table'] = self.signals.table_adder

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(e)
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def reset_random():
    import os
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import warnings
    warnings.filterwarnings('ignore', category=Warning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    import tensorflow as tf
    tf.compat.v1.random.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def plot_line(y1, y2, epochs, for_, save_path):
    fig = plt.figure(num=1)
    plt.plot(range(epochs), y1, label='Training', color='dodgerblue')
    plt.plot(range(epochs), y2, label='Validation', color='orange')
    plt.title('Training and Validation {0}'.format(for_))
    plt.xlabel('Epochs')
    plt.ylabel(for_)
    plt.xlim([0, epochs])
    plt.legend(loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)


def plot_acc_loss(df, path):
    epochs = len(df)
    acc = df['accuracy'].values
    val_acc = df['val_accuracy'].values
    loss = df['loss'].values
    val_loss = df['val_loss'].values
    plot_line(acc, val_acc, epochs, 'Accuracy', path.format('accuracy') + '.png')
    plot_line(loss, val_loss, epochs, 'Loss', path.format('loss') + '.png')


def get_measures(y, pred, prob, log):
    log('Evaluating Performance Measures')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, average='macro')
    recall = recall_score(y, pred, average='macro')
    f1 = f1_score(y, pred, average='macro')
    return list(map(lambda x: round(x, 4), [acc, prec, recall, f1]))


def print_measures_table(df, log):
    import prettytable
    p_table = prettytable.PrettyTable(list(df.columns))
    p_table.add_rows(df.values.tolist())
    log(('\n'.join(['\t{0}'.format(t) for t in p_table.get_string().split('\n')]),))


def plot_cm_roc(y, pred, prob, save_path, log):
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y, pred)
    log('Confusion Matrix')
    log((print_confusion_matrix(conf_mat),))
    log('Plotting Confusion Matrix')
    plot_conf_matrix(conf_mat, 'Confusion Matrix', CLASSES, save_path.format('cm') + '.png')
    log('Plotting ROC Curve')
    plot_roc_curve(y, prob, 'ROC Curve', CLASSES, save_path.format('roc') + '.png')


def plot_conf_matrix(conf_mat, title, labels, path):
    plt.subplots(figsize=(19, 9))
    sbn.heatmap(conf_mat, annot=True, cmap='YlGn', annot_kws={"size": 15}, linewidths=0.5, fmt='d',
                yticklabels=labels, xticklabels=labels)
    plt.xlabel('Predicted Class', labelpad=15)
    plt.ylabel('Actual Class', labelpad=15)
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    plt.clf()


def print_confusion_matrix(cm):
    import prettytable
    import math
    labels = CLASSES
    p_table = prettytable.PrettyTable(['Class'] + labels + ['Total'])
    p_table.align['Class'] = 'l'
    for i, d in enumerate(cm):
        row = [labels[i]]
        row.extend([str(d_).center(8) for d_ in d])
        row.append(d.sum())
        p_table.add_row(row)
    p_table.add_row(['Total'] + cm.sum(axis=0).tolist() + [cm.sum()])
    p_table.hrules = True
    table_string = p_table.get_string().split('\n')
    table_string = [s.replace('-', '\u2500') for s in table_string]
    table_string = [s.replace('|', '\u2502') for s in table_string]
    table_string = [s[:-1] for s in table_string][:-1]
    table_string[-1] = table_string[-1].replace('│', ' ')
    rep = [['\u250C', '\u2510'],
           ['\u251C', '\u2524'],
           ['\u2514', '\u2518']]
    end_idx = table_string[0].rfind('+')
    for r in range(0, len(table_string) - 1, 2):
        extra_space = ' ' * (len(table_string[r]) - end_idx)
        table_string[r] = table_string[r][:end_idx] + extra_space
        if r == 0:
            table_string[r] = rep[0][0] + table_string[r][1:end_idx] + rep[0][1] + extra_space
            table_string[r] = table_string[r].replace('+', '\u252C')
        elif r == len(table_string) - 2:
            table_string[r] = rep[2][0] + table_string[r][1:end_idx] + rep[2][1] + extra_space
            table_string[r] = table_string[r].replace('+', '\u2534')
        else:
            table_string[r] = rep[1][0] + table_string[r][1:end_idx] + rep[1][1] + extra_space
            table_string[r] = table_string[r].replace('+', '\u253C')
    table_string.insert(0, 'Predicted'.center(end_idx))
    add_idx = math.ceil((len(table_string) - 2) / 2)
    for i, s in enumerate(table_string):
        if i == add_idx:
            table_string[i] = 'Actual ' + s
        else:
            table_string[i] = ' ' * 7 + s
    return '\n'.join(['\t{0}'.format(t) for t in table_string])


def plot_roc_curve(y, prob, title, labels, path):
    n_classes = len(labels)
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y, prob[:, i],
                                      pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    y_true = label_binarize(y, classes=range(n_classes))
    y_true = np.hstack((1 - y_true, y_true))
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(),
                                              prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    fig, ax = plt.subplots(1, 1)
    plt.title(title)
    for i in range(n_classes):
        color = plt.cm.get_cmap('tab20')(float(i) / n_classes)
        ax.plot(fpr[i], tpr[i], lw=2, color=color,
                label='{0} (area = {1:0.2f})'
                      ''.format(labels[i], roc_auc[i]))
    ax.plot(fpr['micro'], tpr['micro'],
            label='Micro-Average '
                  '(area = {0:0.2f})'.format(roc_auc['micro']),
            color='deeppink', linestyle=':', linewidth=2)
    ax.plot(fpr['macro'], tpr['macro'],
            label='Macro-Average '
                  '(area = {0:0.2f})'.format(roc_auc['macro']),
            color='navy', linestyle=':', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Threshold == 0.5')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.tick_params()
    ax.legend(loc='lower right', ncol=2)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    plt.clf()
