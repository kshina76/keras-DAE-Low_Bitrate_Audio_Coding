from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import datetime 
from time import time 
import os 
import random
from six.moves import xrange
from tabulate import tabulate 
import scipy.io.wavfile as pywav
from scipy import signal 
from tensorboardX import SummaryWriter
from libs.ConvAE import CAE

def normalize_array(nparr):
    peak = max(abs(nparr.max()), abs(nparr.min()))
    return nparr.astype(np.float32)/peak

def get_dataset(dataset_path, downsample_ratio=0):
    data = {c:[] for c in os.listdir(dataset_path)}
    for classname, wavlist in data.iteritems():
        class_path = os.path.join(dataset_path, classname)
        audio_files = filter(lambda n: n.endswith(".wav"), os.listdir(class_path))
        for i in xrange(len(audio_files)):
            wav_path = os.path.join(class_path, classname+"."+str(i).zfill(5))
            arr = pywav.read(wav_path+".wav")[1].astype(np.float32)
            arr /= max(abs(arr.max()), abs(arr.min()))
            if downsample_ratio > 0:
                arr = signal.decimate(arr, downsample_ratio).astype(np.float32)
            wavlist.append(arr)
    return data

def split_dataset(dataset, train_cv_test_ratio, classes=None):
    if sum(train_cv_test_ratio)!=1:
        raise RuntimeError("[ERROR] split_dataset: ratios don't add up to 1! ")
    train_subset = {}
    cv_subset = {}
    test_subset = {}
    classes = classes if classes else dataset.keys()
    for classname in classes:
        wavlist = dataset[classname]
        random.shuffle(wavlist)
        l = len(wavlist)
        cv_0 = int(l*train_cv_test_ratio[0])
        test_0 = cv_0+int(l*train_cv_test_ratio[1])
        train_subset[classname] = wavlist[0:cv_0]
        cv_subset[classname] = wavlist[cv_0:test_0]
        test_subset[classname] = wavlist[test_0:]
    return train_subset, cv_subset, test_subset

def get_random_batch(dataset, chunk_size, batch_size):
    labels = [random.choice(dataset.keys()) for _ in xrange(batch_size)]
    max_per_class = {cl : len(wavlist)-1 for cl, wavlist in dataset.iteritems()}
    wav_ids = [random.randint(0, max_per_class[l]) for l in labels]
    lengths = [dataset[labels[x]][wav_ids[x]].shape[0] for x in xrange(batch_size)]
    start_idxs = [random.randint(0, lengths[x]-chunk_size) for x in xrange(batch_size)]
    data = np.stack([dataset[labels[x]][wav_ids[x]][start_idxs[x]:start_idxs[x]+chunk_size] for x in xrange(batch_size)])
    return data, labels

def get_class_batch(dataset,clss, chunk_size):
    wav_list = dataset[clss]
    wav_chunks = np.stack([w[x:x+chunk_size] for w in wav_list
                           for x in xrange(0, w.shape[0]-(w.shape[0]%chunk_size),chunk_size)])
    return wav_chunks

class ConfusionMatrix(object):
    def __init__(self, class_domain, name=""):
        self.matrix = {c:{c:0 for c in class_domain} for c in class_domain}
        self.name = name

    def add(self, predictions, labels):
        pred_lbl = zip(predictions, labels)
        for pred, lbl in pred_lbl:
            self.matrix[lbl][pred] += 1

    def __str__(self):
        acc, acc_by_class = self.accuracy()
        classes = sorted(self.matrix.keys())
        short_classes = {c: c[0:8]+"..." if len(c)>8 else c for c in classes}
        prettymatrix = tabulate([[short_classes[c1]]+[self.matrix[c1][c2]
                                 for c2 in classes]+
                                 [acc_by_class[c1]]
                                 for c1 in classes],
                                headers=["real(row)|predicted(col)"]+
                                [short_classes[c] for c in classes]+
                                ["acc. by class"])
        return ("\n"+self.name+" CONFUSION MATRIX\n"+prettymatrix+
                "\n"+self.name+" ACCURACY="+str(acc)+"\n")

    def accuracy(self):
        total = 0 
        diagonal = 0 
        by_class = {c: [0,0] for c in self.matrix}
        acc = float("nan")
        by_class_acc = {c:float("nan") for c in self.matrix}
        for clss, preds in self.matrix.iteritems():
            for pred, n in preds.iteritems():
                total += n
                by_class[clss][1] += n
                if clss==pred:
                    diagonal += n
                    by_class[clss][0] += n
        try:
            acc = float(diagonal)/total
        except ZeroDivisionError:
            pass
        for c,x in by_class.iteritems():
            try:
                by_class_acc[c] = float(x[0])/x[1]
            except ZeroDivisionError:
                pass
        return acc, by_class_acc

def make_graph(model, chunk_shape, num_classes, l2reg=0,
               optimizer_fn=lambda:tf.train.AdamOptimizer()):

    with tf.Graph().as_default() as g:
        data_ph = tf.placeholder(tf.float32, shape=((None,)+chunk_shape),
                                 name="data_placeholder")
        labels_ph = tf.placeholder(tf.int32, shape=(None),
                                   name="labels_placeholder")
        logits, l2nodes = model(data_ph, num_classes)
        predictions = tf.argmax(logits, 1, output_type=tf.int32, name="preds")
        loss = tf.reduce_mean(softmax(logits=logits, labels=labels_ph))
        if l2reg>0:
            loss += l2reg*l2nodes
        global_step = tf.Variable(0, name="global_step", trainable=False)
        minimizer = optimizer_fn().minimize(loss, global_step=global_step)
        inputs = [data_ph, labels_ph]
        outputs = [logits, loss, global_step, minimizer, predictions]
        return g, inputs, outputs

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def make_timestamp2():

    return '{:%d_%b_%Y_%Hh%Mm%Ss}'.format(datetime.datetime.now())

def run_training(train_subset, cv_subset, test_subset, model,
                 batch_size, chunk_size, make_timestamp, max_steps=float("inf"),
                 l2reg=0, optimizer_fn='optim.Adam',
                 train_freq=10, cv_freq=100,
                 save_path=None):
    logger = SummaryWriter('/home/shared/sagnik/output/tensorboard/AE_MusicDenoising/' + \
                            make_timestamp + '_GTZAN_ConvAE')
    
    logger.add_text("Session info", make_timestamp +
                    " model=%s batchsz=%d chunksz=%d l2reg=%f" %
                    (model.__name__, batch_size, chunk_size, l2reg), 0)
    model = CAE()
    model = torch.nn.DataParallel(model)
    model.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001,\
                               betas = (0.9, 0.999), \
                               eps = 1e-08, weight_decay = l2reg)
    train_losses = AverageMeter()
    cv_losses_total = AverageMeter()
    for step in xrange(max_steps):
      data_batch, label_batch = get_random_batch(train_subset, chunk_size,
                                                 batch_size)
      data_batch = data_batch.reshape(data_batch.shape[0],1,-1,1)
      data_batch_torch = torch.from_numpy(data_batch)
      lbl = [CLASS2INT[l] for l in label_batch] 
      input_var = torch.autograd.Variable(data_batch_torch).cuda()
      target_var = torch.autograd.Variable(data_batch_torch).cuda()
      output = model(input_var)
      if step == 0:
        loss = criterion(output, target_var)
      else:
        loss.data = loss.data + criterion(output, target_var)
      train_losses.update(criterion(output, target_var).data[0], data_batch_torch.size(0))
      if(step%train_freq==0):
        model.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      if((step-1)%train_freq == 0):
        print('average training loss after {} training:{}'.format(step, train_losses.avg))
      if(step%cv_freq==0):
        model.eval()
        cv_losses = AverageMeter()
        for c_cv in train_subset:
          print("validating class %s: this may take a while..."%c_cv)
          cv_data = get_class_batch(cv_subset, c_cv, chunk_size)
          cv_data = cv_data.reshape(-1,1, data_batch.shape[1],1)
          cv_data_torch = torch.from_numpy(cv_data)
          cv_labels = cv_data_torch
          input_var = Variable(cv_data_torch).cuda()
          target_var = Variable(cv_labels).cuda()
          output = model(input_var)
          loss = criterion(output, target_var)
          cv_losses.update(loss, cv_data_torch.size(0))
          cv_losses_total.update(loss, cv_data_torch.size(0))
          print('average loss after validating for class:{} is {}'.format(c_cv ,loss/float(cv_data_torch.size(0))))
      if((step - 1)%cv_freq == 0):
        print('Average total validation losses after {} steps of validation:{}'.format(step, cv_losses_total.avg))
    print('final average training loss:{} and final average validation loss:{}'.\
          format(train_losses.avg, cv_losses_total.avg))

class TrainedModel(object):
    def __init__(self, chunk_size, savepath, model_name="my_model"):

        self.chunk_size = chunk_size
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        tf.saved_model.loader.load(self.sess, [model_name], savepath)
        self.data_ph = self.g.get_tensor_by_name("data_placeholder:0")
        self.predictions = self.g.get_tensor_by_name("preds:0")

    def run(self, data_batch):
        return self.sess.run(self.predictions, feed_dict={self.data_ph:data_batch})

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
            return False
        self.close()

DATASET_PATH = "/home/shared/sagnik/datasets/gtzan_reduced/"                                              
TRAIN_CV_TEST_RATIO = [0.5, 0.5, 0.0]
CLASSES =  ["reggae", "classical", "country", "jazz", "metal", "pop", "disco",
            "hiphop", "rock", "blues"]
CLASS2INT = {"reggae":0, "classical":1, "country":2, "jazz":3, "metal":4,
             "pop":5, "disco":6, "hiphop":7, "rock":8, "blues":9}
INT2CLASS = {v:k for k,v in CLASS2INT.iteritems()}
DOWNSAMPLE= 7
BATCH_SIZE = 1000
CHUNK_SIZE = (22050*2)//DOWNSAMPLE
MAX_STEPS=1001
L2_REG = 1e-3

OPTIMIZER_FN = 'optim.Adam' 

TRAIN_FREQ=10
CV_FREQ=100

def run_pretrained_model(data_subset, savepath, chunk_size, batch_size, iters):
    with TrainedModel(chunk_size, savepath) as m:
        cm = ConfusionMatrix(data_subset.keys(), "RELOADED")
        t = time()
        for i in xrange(iters):
            data_batch, label_batch = get_random_batch(data_subset,
                                                       chunk_size, batch_size)
            predictions = m.run(data_batch)
            if (i%100==0):
                print("reloaded model ran", i, "times")
                cm.add([INT2CLASS[x] for x in predictions], label_batch)
        print(cm)
        print("elapsed time:", time()-t)

def main():
    DATA = get_dataset(DATASET_PATH, downsample_ratio=DOWNSAMPLE)
    TRAIN_SUBSET,CV_SUBSET,TEST_SUBSET = split_dataset(DATA,TRAIN_CV_TEST_RATIO,
                                                       CLASSES)
    del DATA
    make_timestamp = make_timestamp2()

    SAVE_PATH = os.path.join("/home/shared/sagnik/output/saved_models", make_timestamp)

    run_training(TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET, MODEL,
                 BATCH_SIZE, CHUNK_SIZE, make_timestamp, MAX_STEPS,
                 L2_REG, OPTIMIZER_FN,
                 TRAIN_FREQ, CV_FREQ,
                 save_path=SAVE_PATH)
    test_pretrained_model(TRAIN_SUBSET, SAVE_PATH,
                          CHUNK_SIZE, 1,50000)
    
main()
