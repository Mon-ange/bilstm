import os
import string
from utils.cnews_loader import build_vocab,process_file,read_category,read_vocab,read_file,batch_iter, to_words
from config.BiLstmTextRelation import BiLstmTextRelation
from config.LSTMConfig import LSTMConfig as config
import tensorflow as tf
import random
import numpy as np
from sklearn import metrics
from pandas import DataFrame
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import codecs
BASE_DIR = 'data'
VOCAB_DIR = os.path.join(BASE_DIR, 'cnews.vocab.txt')
TEST_DIR = os.path.join(BASE_DIR, 'test_data.csv')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation_data.csv')
TRAIN_DIR = os.path.join(BASE_DIR, 'train_data.csv')
SAVE_DIR = 'checkpoints/lstm'
SAVE_PATH = os.path.join(SAVE_DIR, 'best_validation') 
TEST_PREDICTION_PATH='prediction/test_prediction.csv'
vocab_size = 20000
best_accuracy = 0.0

def data_process():
    """
    需要将原数据处理为三个文件夹train、test、val
    :return:
    """
    file_path = r'data/dataset_triage.csv'
    all_list = []

    csvFile = codecs.open(file_path,"r",'utf-8')
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        all_list.append(item[2] + '\t' + item[1] + '\n')


    csvFile.close()

    random.shuffle(all_list)       # 打乱顺序
    test_ratio, val_ratio = 0.1, 0.2

    test_set_size=int(len(all_list)*test_ratio)
    val_set_size=int(len(all_list)*(test_ratio+val_ratio))
    test_set =all_list[:test_set_size]
    val_set =all_list[test_set_size:val_set_size]
    train_set=all_list[val_set_size:]
    with open(TEST_DIR, 'w',encoding='utf8') as w:
        w.writelines(test_set)
    with open(VALIDATION_DIR, 'w',encoding='utf8') as w:
        w.writelines(val_set)
    with open(TRAIN_DIR, 'w',encoding='utf8') as w:
        w.writelines(train_set)

def train():
    trainX, trainY = process_file(TRAIN_DIR, word2id, category2id, config.sequence_length)
    print(trainY)
    print("===sequence_length======",config.sequence_length)
    #2.create session.
    sess_config=tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # 配置 Saver
    with tf.Session() as sess:
        #Instantiate Model
        biLstmTR = BiLstmTextRelation(config.num_classes, config.learning_rate, config.batch_size, config.decay_steps,
                                     config.decay_rate, config.sequence_length,vocab_size, config.embed_size, config.is_training)
        saver = tf.train.Saver()
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        sess.run(init_op)
        curr_epoch=sess.run(biLstmTR.epoch_step)
        # 配置 Saver
        saver = tf.train.Saver()
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=config.batch_size
        for epoch in range(curr_epoch,config.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                curr_loss,curr_acc,_=sess.run([biLstmTR.loss_val,biLstmTR.accuracy,biLstmTR.train_op],
                                              feed_dict={biLstmTR.input_x:trainX[start:end],biLstmTR.input_y:trainY[start:end]
                                              ,biLstmTR.dropout_keep_prob:1.0}) #curr_acc--->TextCNN.accuracy -->,textRNN.dropout_keep_prob:1
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %100==0:# 每100次输出一次
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
                    if(acc > best_accuracy):
                        saver.save(sess=sess,save_path=SAVE_PATH)
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(biLstmTR.epoch_increment)
    pass

def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess,model, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss_val, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def test():
    print("Loading test data...")
    x_test, y_test = process_file(TEST_DIR, word2id, category2id, config.sequence_length)
    biLstmTR = BiLstmTextRelation(config.num_classes, config.learning_rate, config.batch_size, config.decay_steps,
                                     config.decay_rate, config.sequence_length,vocab_size, config.embed_size, config.is_training)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=SAVE_PATH)  # 读取保存的模型
    print('Testing...')
    loss_test, acc_test = evaluate(session, biLstmTR, x_test, y_test)
    msg = 'Test Loss: {0}, Test Acc: {1}'
    print(msg.format(loss_test, acc_test))
    y_test_cls = y_test #np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            biLstmTR.input_x: x_test[start_id:end_id],
            biLstmTR.dropout_keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(biLstmTR.predictions, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    #print(len(categories))
    #print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories,zero_division=0))
    descriptions = np.empty(len(y_pred_cls),dtype=object)
    predictions = np.empty(len(y_pred_cls),dtype=object)
    true_result = np.empty(len(y_pred_cls),dtype=object)
    print(len(descriptions))
    for i in range(len(y_pred_cls)):
        x_test_str = to_words(x_test[i],id2word)
        y_pred_str = categories[y_pred_cls[i]]
        y_test_str = categories[y_test_cls[i]]
        descriptions[i] = x_test_str
        predictions[i] = y_pred_str
        true_result[i] = y_test_str
    data_frame = DataFrame({'descriptions':descriptions,'predictions':predictions,'trueResule':true_result})
    data_frame.to_csv(TEST_PREDICTION_PATH,index=False)

operation = "test"

if __name__ == "__main__":
    #print("hi")
    data_process()
    if not os.path.exists(VOCAB_DIR):  # 如果不存在词汇表，重建
        build_vocab(TRAIN_DIR, VOCAB_DIR, vocab_size)
    categories, category2id = read_category()
    words, word2id, id2word = read_vocab(VOCAB_DIR)
    vocab_size = len(words)
    if(operation == "train"):
        train()
    if(operation == "test"):
        test()