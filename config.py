class gen_config(object):


    seed = 3435
    batch_size = 5  # batch size
    num_workers = 2  # how many workers for loading data
    max_len = 80 + 2  # max_len for each sentence + two padding
    limit = 50  # the position range <-limit, limit>
    vocab_size = 160695+2
    rel_num = 30
    type_num =2 # true or false
    word_dim = 50
    pos_dim = 5
    pos_size = limit * 2 + 2



    n_epochs = 16  # the number of epochs for training
    epoch_per_test = 5
    drop_out = 0.5
    lr = 0.0003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter
    opt_method = 'SGD'


    norm_emb = True
    use_pcnn = False
    use_cuda = True

    # Conv
    filters = [3]
    filters_num = 230 #
    sen_feature_dim = filters_num

    rel_dim = filters_num * len(filters)
    rel_filters_num = 100

    model_name = 'generator.pkl'
    w2v_path = 'data/dic_data/w2v.npy'
    p1_2v_path = 'data/dic_data/p1_2v.npy'
    p2_2v_path = 'data/dic_data/p2_2v.npy'


class dis_config():
    seed = 3435
    batch_size = 5  # batch size
    num_workers = 2  # how many workers for loading data
    max_len = 80 + 2  # max_len for each sentence + two padding
    limit = 50  # the position range <-limit, limit>
    vocab_size = 160695 + 2
    rel_num = 30
    type_num = 2  # true or false
    word_dim = 50
    pos_dim = 5
    pos_size = limit * 2 + 2

    n_epochs = 16  # the number of epochs for training
    epoch_per_test = 5
    drop_out = 0.5
    lr = 0.0003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter
    opt_method = 'SGD'

    norm_emb = True
    use_pcnn = False
    use_cuda = True

    # Conv
    filters = [3]
    filters_num = 230  #
    sen_feature_dim = filters_num

    rel_dim = filters_num * len (filters)
    rel_filters_num = 100

    model_name = 'discriminator.pkl'
    w2v_path = 'data/dic_data/w2v.npy'
    p1_2v_path = 'data/dic_data/p1_2v.npy'
    p2_2v_path = 'data/dic_data/p2_2v.npy'

class gan_config():
    seed = 3435
    batch_size = 5  # batch size
    num_workers = 2  # how many workers for loading data
    max_len = 80 + 2  # max_len for each sentence + two padding
    limit = 50  # the position range <-limit, limit>
    vocab_size = 160695 + 2
    rel_num = 30
    type_num = 2  # true or false
    word_dim = 50
    pos_dim = 5
    pos_size = limit * 2 + 2

    n_epochs = 16  # the number of epochs for training
    epoch_per_test = 5
    drop_out = 0.5
    lr = 0.0003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter
    opt_method = 'SGD'

    norm_emb = True
    use_pcnn = False
    use_cuda = True

    # Conv
    filters = [3]
    filters_num = 230  #
    sen_feature_dim = filters_num

    rel_dim = filters_num * len (filters)
    rel_filters_num = 100

    model_name = 'gan.pkl'
    w2v_path = 'data/dic_data/w2v.npy'
    p1_2v_path = 'data/dic_data/p1_2v.npy'
    p2_2v_path = 'data/dic_data/p2_2v.npy'


