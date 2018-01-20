
# the dpn 68, user larsw , like dood_pool + lars

import mxnet as mx

bn_momentum = 0.9

def BK(data):
    return mx.symbol.BlockGrad(data=data)

# - - - - - - - - - - - - - - - - - - - - - - -
# Fundamental Elements
def BN(data, fix_gamma=False, momentum=bn_momentum, name=None):
    #bn     = mx.symbol.BatchNorm( data=data, fix_gamma=fix_gamma, momentum=bn_momentum, name=('%s__bn'%name))
    bn     = mx.symbol.BatchNorm( data=data, name=('%s__bn'%name))
    return bn

def AC(data, act_type='relu', name=None):
    act    = mx.symbol.Activation(data=data, act_type=act_type, name=('%s__%s' % (name, act_type)))
    return act

def BN_AC(data, momentum=bn_momentum, name=None):
    bn     = BN(data=data, name=name, fix_gamma=False, momentum=momentum)
    bn_ac  = AC(data=bn,   name=name)
    return bn_ac

def Conv(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, no_bias=True, w=None, b=None, attr=None, num_group=1):
    Convolution = mx.symbol.Convolution
    if w is None:
        conv     = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, attr=attr)
    else:
        if b is None:
            conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, weight=w, attr=attr)
        else:
            conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=False, bias=b, weight=w, attr=attr)
    return conv

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < CVPR >
def Conv_BN(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov    = Conv(   data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_bn = BN(     data=cov,    name=('%s__bn' % name))
    return cov_bn

def Conv_BN_AC(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov_bn = Conv_BN(data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_ba = AC(     data=cov_bn, name=('%s__ac' % name))
    return cov_ba

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < ECCV >
def BN_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    bn_cov = Conv(   data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return bn_cov

def AC_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    ac     = AC(     data=data,   name=('%s__ac' % name))
    ac_cov = Conv(   data=ac,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return ac_cov

def BN_AC_Conv(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    ba_cov = AC_Conv(data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return ba_cov


# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Dual Path Unit
def DualPathFactory(data, num_1x1_a, num_3x3_b, num_1x1_c, name, inc, G, _type='normal'):
    kw = 3
    kh = 3
    pw = (kw-1)/2
    ph = (kh-1)/2

    # type
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    # PROJ
    if type(data) is list:
        data_in  = mx.symbol.Concat(*[data[0], data[1]],  name=('%s_cat-input' % name))
    else:
        data_in  = data

    if has_proj:
        c1x1_w   = BN_AC_Conv( data=data_in, num_filter=(num_1x1_c+2*inc), kernel=( 1, 1), stride=(key_stride, key_stride), name=('%s_c1x1-w(s/%d)' %(name, key_stride)), pad=(0, 0))
        data_o1  = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=0,         end=num_1x1_c,         name=('%s_c1x1-w(s/%d)-split1' %(name, key_stride)))
        data_o2  = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=num_1x1_c, end=(num_1x1_c+2*inc), name=('%s_c1x1-w(s/%d)-split2' %(name, key_stride)))
    else:
        data_o1  = data[0]
        data_o2  = data[1]
        
    # MAIN
    c1x1_a = BN_AC_Conv( data=data_in, num_filter=num_1x1_a,       kernel=( 1,  1), pad=( 0,  0), name=('%s_c1x1-a'   % name))
    c3x3_b = BN_AC_Conv( data=c1x1_a,  num_filter=num_3x3_b,       kernel=(kw, kh), pad=(pw, ph), name=('%s_c%dx%d-b' % (name,kw,kh)), stride=(key_stride,key_stride), num_group=G)
    c1x1_c = BN_AC_Conv( data=c3x3_b,  num_filter=(num_1x1_c+inc), kernel=( 1,  1), pad=( 0,  0), name=('%s_c1x1-c'   % name))
    c1x1_c1= mx.symbol.slice_axis(data=c1x1_c, axis=1, begin=0,         end=num_1x1_c,       name=('%s_c1x1-c-split1' % name))
    c1x1_c2= mx.symbol.slice_axis(data=c1x1_c, axis=1, begin=num_1x1_c, end=(num_1x1_c+inc), name=('%s_c1x1-c-split2' % name))
    print('res chanel %s, ds chanel %s' % (num_1x1_c, inc))

    #c1x1_c = BN_AC( data=c3x3_b,  name=('%s_c1x1-c'  % name))
    #c1x1_c1= Conv(  data=c1x1_c,  num_filter=num_1x1_c, kernel=( 1,  1), name=('%s_c1x1-c1' % name),         pad=( 0,  0))
    #c1x1_c2= Conv(  data=c1x1_c,  num_filter=inc,       kernel=( 1,  1), name=('%s_c1x1-c2' % name),         pad=( 0,  0))

    # OUTPUTS
    summ   = mx.symbol.ElementWiseSum(*[data_o1, c1x1_c1],                        name=('%s_sum' % name))
    summ._set_attr(mirror_stage='True')
    dense  = mx.symbol.Concat(        *[data_o2, c1x1_c2],                        name=('%s_cat' % name))
    dense._set_attr(mirror_stage='True')

    return [summ, dense]



k_R = 128
G   = 32

"""
k_sec  = {  2: 3,  \
            3: 4,  \
            4: 12, \
            5: 3   }

inc_sec= {  2: 16, \
            3: 32, \
            4: 32, \
            5: 64  }
"""

k_sec  = {  2: 3,  \
            3: 4,  \
            4: 4, \
            5: 4   }

inc_sec= {  2: 16, \
            3: 32, \
            4: 64, \
            5: 128  }

def get_before_pool():

    ## define Dual Path Network
    data = mx.symbol.Variable(name="data")

    # conv1
    conv1_x_1  = Conv(data=data,  num_filter=64,  kernel=(3, 3), name='conv1_x_1', pad=(1,1), stride=(2,2))
    conv1_x_1  = BN_AC(conv1_x_1, name='conv1_x_1__relu-sp')
    conv1_x_x  = mx.symbol.Pooling(data=conv1_x_1, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    # conv2
    bw = 64
    inc= inc_sec[2]
    R  = (k_R*bw)/64 
    conv2_x_x  = DualPathFactory(     conv1_x_x,   R,   R,   bw,  'conv2_x__1',           inc,   G,  'proj'  )
    for i_ly in range(2, k_sec[2]+1):
        inc += 48 
        conv2_x_x  = DualPathFactory( conv2_x_x,   R,   R,   bw, ('conv2_x__%d'% i_ly),   inc,   G,  'normal')

    # conv3
    bw = 128
    inc= inc_sec[3]
    R  = (k_R*bw)/64
    conv3_x_x  = DualPathFactory(     conv2_x_x,   R,   R,   bw,  'conv3_x__1',           inc,   G,  'down'  )
    for i_ly in range(2, k_sec[3]+1):
        inc += 48
        conv3_x_x  = DualPathFactory( conv3_x_x,   R,   R,   bw, ('conv3_x__%d'% i_ly),   inc,   G,  'normal')

    # conv4
    bw = 256
    inc= inc_sec[4]
    R  = (k_R*bw)/64
    conv4_x_x  = DualPathFactory(     conv3_x_x,   R,   R,   bw,  'conv4_x__1',           inc,   G,  'down'  )
    for i_ly in range(2, k_sec[4]+1):
        inc += 48
        conv4_x_x  = DualPathFactory( conv4_x_x,   R,   R,   bw, ('conv4_x__%d'% i_ly),   inc,   G,  'normal')

    # conv5
    bw = 512
    inc= inc_sec[5]
    R  = (k_R*bw)/64
    conv5_x_x  = DualPathFactory(     conv4_x_x,   R,   R,   bw,  'conv5_x__1',           inc,   G,  'down'  )
    for i_ly in range(2, k_sec[5]+1):
        inc += 48
        conv5_x_x  = DualPathFactory( conv5_x_x,   R,   R,   bw, ('conv5_x__%d'% i_ly),   inc,   G,  'normal')

    # output: concat
    conv5_x_x  = mx.symbol.Concat(*[conv5_x_x[0], conv5_x_x[1]],  name='conv5_x_x_cat-final')
    #conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
    return conv5_x_x

def get_linear(num_classes = 1000):
    before_pool = get_before_pool()
    # - - - - -
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="pool5")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='fc6')
    return fc6

def get_symbol_old(num_classes = 1000):
    fc6       = get_linear(num_classes)
    softmax   = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    sys_out   = softmax
    return sys_out

def get_symbol(num_classes, feature_dim, is_finetune, **net_params):
    num_fc_feature = net_params.get('num_fc_feature', None)
    dropout = False

    before_pool = get_before_pool()
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="pool5")
    flat = mx.symbol.Flatten(data=pool5, name='flatten')

    #backbone, nchannels = Dsod_backbone(feature_dim, num_fc_feature, dropout, final_channels=256)
    #flat = mx.symbol.Flatten(data=backbone)

    if num_fc_feature is None:
        return mx.symbol.FullyConnected(data=flat, num_hidden=feature_dim, name='fc1')
    else:
        fc_list = []
        assert(feature_dim % num_fc_feature == 0)
        num_hidden = feature_dim / num_fc_feature
        for i in range(num_fc_feature):
            fc = mx.symbol.FullyConnected(data=flat, num_hidden=num_hidden, name='fc1_block_%d' % i)
            fc_list.append(fc)
        concat_feature = mx.symbol.Concat(*fc_list, dim=1, name='fc1')
        return concat_feature
