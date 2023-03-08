class LSTMConfig(object):
    num_classes = 1056 #number of label
    learning_rate = 0.1 #learning rate
    batch_size = 32 #Batch size for training/evaluating. 批处理的大小 32-->128
    decay_steps = 12000 #how many steps before decay learning rate. 批处理的大小 32-->128
    decay_rate = 0.9 #Rate of decay for learning rate 0.5一次衰减多少
    ckpt_dir = "biLstm_text_relation_checkpoint/" #checkpoint location for the model
    sequence_length = 400 #max sentence length
    embed_size = 80 #embedding size
    is_training = True #is traing.true:training,false:testing/inference
    num_epochs = 120 # 
    validate_every = 10 #Validate every validate_every epochs 每10轮做一次验证
    use_embedding = True #whether to use embedding or not
    training_data_path = "/home/xul/xul/9_ZhihuCup/test_twoCNN_zhihu.txt"#path of traning data. train-zhihu4-only-title-all.txt===>training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
    word2vec_model_path = "zhihu-word2vec.bin-100" #word2vec's vocabulary and vectors
    