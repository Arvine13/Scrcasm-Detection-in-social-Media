if True:
    from utils import reset_random, PandasDfToPyqtTable, Worker, plot_acc_loss, plot_cm_roc, get_measures, \
        print_measures_table

    reset_random()
import json
import os.path
import pickle
import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QGroupBox, QGridLayout, QScrollArea, \
    QPushButton, QCheckBox, QPlainTextEdit, QFrame, QDialog, QProgressBar, QTableView, QAbstractItemView
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.utils.np_utils import to_categorical

from cnn_rnn import cnn_rnn
from feature_engineering import get_tokenized_data, MAX_LENGTH, get_glove_data, EMBEDDING_DIM
from preprocess import preprocess_df
from utils import clear_layout


def parse_data(path, log):
    log.emit('Loading Data From :: {0}'.format(path))
    with open(path, 'r') as file:
        for line in file.readlines():
            yield json.loads(line)


def json_to_df(json_path, log):
    json_data = list(parse_data(json_path, log))
    df = pd.DataFrame(pd.json_normalize(json_data), columns=['headline', 'is_sarcastic'])
    return df


class TrainingCallback(Callback):
    def __init__(self, acc_loss_csv_path_, acc_loss_graph_path_, log):
        self.acc_loss_csv_path = acc_loss_csv_path_
        self.acc_loss_graph_path_ = acc_loss_graph_path_
        self.log = log
        if os.path.isfile(self.acc_loss_csv_path):
            self.df = pd.read_csv(self.acc_loss_csv_path)
        else:
            self.df = pd.DataFrame([], columns=['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss'])
            self.df.to_csv(self.acc_loss_csv_path, index=False)
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs=None):
        self.df.loc[len(self.df.index)] = [
            str(int(epoch + 1)).zfill(2), round(logs['accuracy'], 4), round(logs['val_accuracy'], 4),
            round(logs['loss'], 4), round(logs['val_loss'], 4)
        ]
        self.df.to_csv(self.acc_loss_csv_path, index=False)
        self.log('[EPOCH :: {0}] -> Acc :: {1} | Val_Acc :: {2} | Loss :: {2} | Val_Loss :: {3}'.format(
            *[self.df.values[-1][0]], *[str(v).ljust(6, '0') for v in self.df.values[-1][1:]])
        )
        plot_acc_loss(self.df, self.acc_loss_graph_path_)


class MainGUI(QWidget):
    def __init__(self):
        super(MainGUI, self).__init__()
        self.screen_size = app.primaryScreen().availableSize()
        self.app_width = self.screen_size.width()
        self.app_height = self.screen_size.height()
        self.setWindowTitle('Sarcasm Detection With TLBO')
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        self.full_h_box = QHBoxLayout()
        self.full_h_box.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.left_v_box = QVBoxLayout()
        self.right_v_box = QVBoxLayout()

        self.gb_1 = QGroupBox('Input')
        self.gb_1.setFixedWidth((self.app_width // 100) * 49)
        self.gb_1.setFixedHeight((self.app_height // 100) * 20)
        self.grid_1 = QGridLayout()
        self.grid_1.setSpacing(20)
        self.gb_1.setLayout(self.grid_1)

        self.load_data_btn = QPushButton('Load Data')
        self.load_data_btn.clicked.connect(self.load_dataset_thread)
        self.grid_1.addWidget(self.load_data_btn, 0, 0)

        self.preprocess_btn = QPushButton('PreProcess')
        self.preprocess_btn.clicked.connect(self.preprocess_thread)
        self.grid_1.addWidget(self.preprocess_btn, 0, 1)

        self.tlbo_cb = QCheckBox('Teaching-Learning Based Optimization')
        self.grid_1.addWidget(self.tlbo_cb, 0, 2, Qt.AlignRight)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset)
        self.grid_1.addWidget(self.reset_btn, 1, 0)

        self.fe_btn = QPushButton('Feature Engineering')
        self.fe_btn.clicked.connect(self.feature_engineering_thread)
        self.grid_1.addWidget(self.fe_btn, 1, 1)

        self.train_btn = QPushButton('Train CNN+RNN(LSTM) Network')
        self.train_btn.clicked.connect(self.train_cnn_rnn_lstm_thread)
        self.grid_1.addWidget(self.train_btn, 1, 2)

        self.gb_2 = QGroupBox('Data')
        self.gb_2.setFixedWidth((self.app_width // 100) * 49)
        self.gb_2.setFixedHeight((self.app_height // 100) * 78)

        self.grid_2_scroll = QScrollArea()
        self.grid_2_scroll.setFrameShape(False)
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.grid_2_widget.hide()

        self.grid_2 = QGridLayout(self.grid_2_widget)
        self.gb_2.setLayout(self.gb_2_v_box)
        self.grid_2.setSpacing(20)
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)

        self.gb_3 = QGroupBox('Process')
        self.gb_3.setFixedWidth((self.app_width // 100) * 49)
        self.gb_3.setFixedHeight((self.app_height // 100) * 99)
        self.grid_3 = QGridLayout()
        self.grid_3.setSpacing(20)
        self.gb_3.setLayout(self.grid_3)

        self.process_pte = QPlainTextEdit()
        self.process_pte.setFont(QFont('JetBrains Mono', 10))
        self.process_pte.setStyleSheet('background-color: transparent;')
        self.process_pte.setReadOnly(True)
        self.process_pte.setFrameShape(QFrame.NoFrame)
        self.grid_3.addWidget(self.process_pte, 0, 1)

        self.full_h_box.addLayout(self.left_v_box)
        self.left_v_box.addWidget(self.gb_1)
        self.left_v_box.addWidget(self.gb_2)
        self.full_h_box.addLayout(self.right_v_box)
        self.right_v_box.addWidget(self.gb_3)
        self.setLayout(self.full_h_box)

        self.source_df = None
        self.preprocessed_df = None
        self.train_df = None
        self.vocab_size = None
        self.embedding_matrix = None
        self.load_screen = Loading()
        self.thread_pool = QThreadPool()

        self.showMaximized()

    def update_log(self, text):
        if isinstance(text, str):
            self.process_pte.appendPlainText('>>> {0}'.format(text))
        elif isinstance(text, tuple):
            self.process_pte.appendPlainText(text[0])

    def add_table(self, df):
        tableView = QTableView(self)
        model = PandasDfToPyqtTable(df)
        tableView.setFixedWidth((self.gb_2.width() // 100) * 100)
        tableView.setModel(model)
        tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tableView.resizeColumnsToContents()
        self.grid_2.addWidget(tableView, self.grid_2.count() + 1, 0, Qt.AlignHCenter)

    def load_dataset_thread(self):
        self.reset()
        worker = Worker(self.load_dataset_runner)
        worker.signals.finished.connect(self.load_dataset_finisher)
        worker.signals.log_updater.connect(self.update_log)
        worker.signals.table_adder.connect(self.add_table)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.load_data_btn.setEnabled(False)
        self.thread_pool.start(worker)

    def load_dataset_runner(self, log, table):
        df1 = json_to_df('Data/source/Sarcasm_Headlines_Dataset.json', log)
        df2 = json_to_df('Data/source/Sarcasm_Headlines_Dataset_v2.json', log)
        log.emit('Concatenating DataFrames')
        df_ = pd.concat([df1, df2])
        log.emit('Data Shape :: {0}'.format(df_.shape))

        log.emit('Dropping Duplicates')
        df_.drop_duplicates(inplace=True)
        log.emit('Data Shape After Dropping Duplicates :: {0}'.format(df_.shape))

        data_save_path = 'Data/sarcasm.csv'
        log.emit('Saving Data :: {0}'.format(data_save_path))
        df_.to_csv(data_save_path, index=False)
        self.source_df = df_.copy(deep=True)

        log.emit('Data Loaded!')

    def load_dataset_finisher(self):
        self.preprocess_btn.setEnabled(True)
        self.add_table(self.source_df.head(100).copy())
        self.load_screen.close()

    def preprocess_thread(self):
        worker = Worker(self.preprocess_runner)
        worker.signals.finished.connect(self.preprocess_finisher)
        worker.signals.log_updater.connect(self.update_log)
        worker.signals.table_adder.connect(self.add_table)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.thread_pool.start(worker)

    def preprocess_runner(self, log, table):
        df_ = preprocess_df(self.source_df.copy(), 'headline', log.emit)
        df_.dropna(inplace=True)
        df_.to_csv('Data/sarcasm_preprocessed.csv', index=False)
        self.preprocessed_df = df_.copy(deep=True)

    def preprocess_finisher(self):
        self.add_table(self.preprocessed_df.head(100).copy())
        self.preprocess_btn.setEnabled(False)
        self.fe_btn.setEnabled(True)
        self.load_screen.close()

    def feature_engineering_thread(self):
        worker = Worker(self.feature_engineering_runner)
        worker.signals.finished.connect(self.feature_engineering_finisher)
        worker.signals.log_updater.connect(self.update_log)
        worker.signals.table_adder.connect(self.add_table)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.thread_pool.start(worker)

    def feature_engineering_runner(self, log, table):
        data = self.preprocessed_df['headline'].values
        labels = self.preprocessed_df['is_sarcastic'].values
        tokenized_data, word_indices = get_tokenized_data(data, log.emit)
        embedding_matrix = get_glove_data(word_indices, log.emit)
        self.vocab_size = len(word_indices)

        data_path = 'Data/sarcasm_training.csv'
        embedding_path = 'Data/glove_embedding_matrix.pkl'

        log.emit('Saving Data :: {0}'.format(data_path))
        train_df = pd.DataFrame(tokenized_data, columns=range(1, MAX_LENGTH + 1))
        train_df['Sarcastic'] = labels
        train_df.to_csv(data_path, index=False)
        self.train_df = train_df

        log.emit('Saving Glove Embedding Matrix :: {0}'.format(embedding_path))
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding_matrix, f)
        self.embedding_matrix = embedding_matrix

    def feature_engineering_finisher(self):
        self.fe_btn.setEnabled(False)
        self.tlbo_cb.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.load_screen.close()
        self.add_table(self.train_df.head(100).copy())

    def train_cnn_rnn_lstm_thread(self):
        worker = Worker(self.train_cnn_rnn_lstm_runner)
        worker.signals.finished.connect(self.train_cnn_rnn_lstm_finisher)
        worker.signals.log_updater.connect(self.update_log)
        worker.signals.table_adder.connect(self.add_table)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.thread_pool.start(worker)

    def train_cnn_rnn_lstm_runner(self, log, table):
        x_ = self.train_df.values[:, :-1]
        y_ = self.train_df.values[:, -1]
        y_cat = to_categorical(y_, num_classes=2)

        tlbo = self.tlbo_cb.isChecked()

        log.emit('X Shape :: {0}'.format(x_.shape))
        log.emit('Y Shape :: {0}'.format(y_cat.shape))

        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        tlbo_name = 'With{0}_TLBO'.format('' if tlbo else 'out')
        model_path = os.path.join(model_dir, 'model_{0}.h5'.format(tlbo_name))
        acc_loss_csv_path = os.path.join(model_dir, 'acc_loss_{0}.csv'.format(tlbo_name))
        acc_loss_graph_path = os.path.join(model_dir, '{0}_' + tlbo_name)

        training_cb = TrainingCallback(acc_loss_csv_path, acc_loss_graph_path, log.emit)
        checkpoint = ModelCheckpoint(model_path, save_best_only=True,
                                     monitor='val_accuracy', mode='max', verbose=False)
        model = cnn_rnn(self.vocab_size, EMBEDDING_DIM, MAX_LENGTH, self.embedding_matrix, tlbo, log.emit)
        initial_epoch = 0
        if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
            log.emit('Loading Pre-Trained Model :: {0}'.format(model_path))
            model.load_weights(model_path)
            initial_epoch = len(pd.read_csv(acc_loss_csv_path))

        log.emit('Fitting Data')
        model.fit(x_, y_cat, validation_data=(x_, y_cat),
                  callbacks=[checkpoint, training_cb], batch_size=128,
                  epochs=20, initial_epoch=initial_epoch, verbose=0)

        log.emit('Predicting Data')
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        model.load_weights(model_path)
        prob = model.predict(x_)
        pred = np.argmax(prob, axis=1)
        plot_cm_roc(y_, pred, prob, os.path.join(results_dir, '{0}_' + tlbo_name), log.emit)
        measure_names = ['Accuracy', 'Precision', 'Recall', 'F-Measure']
        df = pd.DataFrame([get_measures(y_, pred, prob, log.emit)], columns=measure_names)
        print_measures_table(df, log.emit)
        df.to_csv(os.path.join(results_dir, 'measures_{0}.csv'.format(tlbo_name)), index=False)

    def train_cnn_rnn_lstm_finisher(self):
        self.tlbo_cb.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.load_screen.close()

    def disable(self):
        self.preprocess_btn.setEnabled(False)
        self.fe_btn.setEnabled(False)
        self.tlbo_cb.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.load_data_btn.setEnabled(True)

    def reset(self):
        self.disable()
        clear_layout(self.grid_2)
        self.process_pte.clear()


class Loading(QDialog):
    def __init__(self, parent=None):
        super(Loading, self).__init__(parent)
        self.screen_size = app.primaryScreen().size()
        self._width = (self.screen_size.width() // 100) * 40
        self._height = (self.screen_size.height() // 100) * 5
        self.setGeometry(0, 0, self._width, self._height)
        x = (self.screen_size.width() - self.width()) // 2
        y = (self.screen_size.height() - self.height()) // 2
        self.move(x, y)
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.pb = QProgressBar(self)
        self.pb.setFixedWidth(self.width())
        self.pb.setFixedHeight(self.height())
        self.pb.setRange(0, 0)


if __name__ == '__main__':
    app = QApplication([sys.argv])
    app.setStyle('Fusion')
    app.setFont(QFont('JetBrains Mono'))
    window = MainGUI()
    sys.exit(app.exec_())
