import sys
import numpy as np
import pyaudio
import wave
import torch
import os
import resnest.torch as resnest_torch
from torch import nn
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from test3 import Ui_Dialog
from PyQt5 import QtCore
from pujianfunc import pujian
import spectrogramfunc as sf
import shutil
import yaml
import librosa
import librosa.display
import matplotlib.pyplot as plt

settings_str = """
loader:
  train:
    batch_size: 16
    shuffle: True
  val:
    batch_size: 32
    shuffle: False

model:
  name: resnest50_fast_1s1x64d
  params:
    pretrained: False
    n_classes: 10

loss:
  name: BCEWithLogitsLoss
  params: {}

optimizer:
  name: Adam
  params:
    lr: 0.001
"""
settings = yaml.safe_load(settings_str)


# 此处可能需要填入模型类
def get_model():
    model = resnest_torch.resnest50_fast_1s1x64d(pretrained=False)
    del model.fc
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 10))
    return model


def save_img(wav_path, output_place, figsize):
    # 加载音频文件
    y, sr = librosa.load(wav_path, sr=None)

    # 提取Melspectrogram和MFCC特征
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)

    # 绘制波形图
    plt.figure(figsize=figsize)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.savefig(output_place + '\\' + 'waveform.png', dpi=300, bbox_inches='tight')

    # 绘制Melspectrogram图
    plt.figure(figsize=figsize)
    librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.savefig(output_place + '\\' + 'melspectrogram.png', dpi=300, bbox_inches='tight')

    # 绘制MFCC图
    plt.figure(figsize=figsize)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig(output_place + '\\' + 'mfcc.png', dpi=300, bbox_inches='tight')


class MyMainWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.y = ''
        self.py = "ptest.wav"
        self.des = "data"
        self.upload.clicked.connect(self.FileUpLoad)
        self.recording.clicked.connect(self.GetWav)
        self.test.clicked.connect(self.test_y)
        self.reset.clicked.connect(self.ResetAll)

    def FileUpLoad(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '音频文件(*.wav)')
        # playsound(fname)
        _translate = QtCore.QCoreApplication.translate
        self.y = fname
        self.lineEdit.setText(_translate("Dialog", self.y))
        print(self.y)

    def GetWav(self):
        # 创建对象
        pa = pyaudio.PyAudio()
        # 创建流：采样位，声道数，采样频率，缓冲区大小，input
        stream = pa.open(format=pyaudio.paInt16,
                         channels=2,
                         rate=16000,
                         input=True,
                         frames_per_buffer=1024)
        # 获取当前目录
        current_dir = os.getcwd()
        # 创建保存音频文件的目录
        save_dir = os.path.join(current_dir, 'temp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 创建式打开音频文件
        wf = wave.open(os.path.join(save_dir, 'test.wav'), "wb")
        # 设置音频文件的属性：采样位，声道数，采样频率，缓冲区大小，必须与流的属性一致
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        print("开始录音")
        # 采样频率*秒数/缓冲区大小
        sec = 5
        for w in range(int(16000 * sec / 1024)):
            data = stream.read(1024)  # 每次从流的缓存中读出数据
            wf.writeframes(data)  # 把读出的数据写入wf
        print("录音结束")
        stream.stop_stream()  # 先把流停止
        stream.close()  # 再把流关闭
        pa.terminate()  # 把对象关闭
        wf.close()  # 把声音文件关闭
        fname = os.path.join(save_dir, 'test.wav')
        self.y = fname
        _translate = QtCore.QCoreApplication.translate
        self.lineEdit.setText(_translate("Dialog", self.y))

    def showImage(self, mfcc_path, mel_path, voice_path, birdimg_path):
        frame = QImage(voice_path)
        frame_size = frame.size()
        view_size = self.sound.viewport().size()
        # 计算缩放比例
        scale_factor = min(view_size.width() / frame_size.width(),
                           view_size.height() / frame_size.height())
        # 创建缩放后的图像
        scaled_frame = frame.scaled(frame_size * scale_factor)
        pix = QPixmap.fromImage(scaled_frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.sound.setScene(scene)

        frame = QImage(mfcc_path)
        frame_size = frame.size()
        view_size = self.sound.viewport().size()
        # 计算缩放比例
        scale_factor = min(view_size.width() / frame_size.width(),
                           view_size.height() / frame_size.height())
        # 创建缩放后的图像
        scaled_frame = frame.scaled(frame_size * scale_factor)
        pix = QPixmap.fromImage(scaled_frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.mfcc.setScene(scene)

        frame = QImage(mel_path)
        frame_size = frame.size()
        view_size = self.sound.viewport().size()
        # 计算缩放比例
        scale_factor = min(view_size.width() / frame_size.width(),
                           view_size.height() / frame_size.height())
        # 创建缩放后的图像
        scaled_frame = frame.scaled(frame_size * scale_factor)
        pix = QPixmap.fromImage(scaled_frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.Spec.setScene(scene)

        frame = QImage(birdimg_path)
        frame_size = frame.size()
        view_size = self.sound.viewport().size()
        # 计算缩放比例
        scale_factor = min(view_size.width() / frame_size.width(),
                           view_size.height() / frame_size.height())
        # 创建缩放后的图像
        scaled_frame = frame.scaled(frame_size * scale_factor)
        pix = QPixmap.fromImage(scaled_frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.birdimage.setScene(scene)

    def test_y(self):
        dir_path = os.path.join(os.getcwd(), 'temp')
        img_path = os.path.join(os.getcwd(), 'img')
        file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
        # 删除文件
        for file_path in file_paths:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error: {e}')
        file_paths = [os.path.join(img_path, file) for file in os.listdir(img_path)]
        # 删除文件
        for file_path in file_paths:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error: {e}')
        pujian(self.y, dir_path + '\\' + self.py)
        data = sf.trans_spec(dir_path)
        model = get_model()
        model.load_state_dict(torch.load("resnest50birds10.pth"), False)# 加载已保存的模型，可能需要改名字
        model = model.to("cuda")
        model.eval()
        with torch.no_grad():
            output = model(torch.from_numpy(np.expand_dims(data, axis=0)).float().to("cuda"))
        outputlist = output.tolist()
        b = np.argmax(outputlist[0]) + 1
        birdimg_path_0 = os.path.join(os.getcwd(), 'birdimage')
        if b == 1:
            result = '白腰草鹬'
            bird_name = '白腰草鹬'
        elif b == 2:
            result = '黄鹡鸰'
            bird_name = "黄鹡鸰"
        elif b == 3:
            result = '家燕'
            bird_name = "家燕"
        elif b == 4:
            result = '矶鹬'
            bird_name = "矶鹬"
        elif b == 5:
            result = '大杜鹃'
            bird_name = "大杜鹃"
        elif b == 6:
            result = '黑水鸡'
            bird_name = "黑水鸡"
        elif b == 7:
            result = '大杜鹃'
            bird_name = "大杜鹃"
        elif b == 8:
            result = '白腰草鹬'
            bird_name = "白腰草鹬"
        elif b == 9:
            result = '白鹡鸰'
            bird_name = "白鹡鸰"
        elif b == 10:
            result = '红脚鹬'
            bird_name = "红脚鹬"
        else:
            result = 'False'
            bird_name = 'False'

        save_img(dir_path + '\\' + self.py, img_path, figsize=(5, 2))

        self.showImage(img_path + '\\' + 'mfcc', img_path + '\\' + 'melspectrogram',
                       img_path + '\\' + 'waveform', birdimg_path_0 + '\\' + bird_name + '.jpg')

        self.textBrowser.append(result)
        # 输出为标签，后续需要改为对应鸟的名称

    def ResetAll(self):
        # 重置所有变量和界面控件
        self.y = ''
        self.py = "ptest.wav"
        self.des = "data"
        self.lineEdit.clear()
        self.textBrowser.clear()
        self.mfcc.scene().clear()
        self.sound.scene().clear()
        self.Spec.scene().clear()
        self.birdimage.scene().clear()
        # self.your_label.setText('')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
