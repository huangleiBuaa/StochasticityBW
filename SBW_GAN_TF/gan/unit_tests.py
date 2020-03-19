from layers.spectral_normalized_layers import *
from tensorflow.python.keras.models import Model, Input


def test_conv_with_conv_spectal():
    from tensorflow.python.keras.models import Model, Input
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((3, 3, 1))
    out = SNConv2D(kernel_size=(2, 2), padding='valid', filters=1, kernel_initializer=kernel_init, stateful=True, conv_singular=True)(inp)
    m = Model([inp], [out])
    x = np.arange(3 * 3).reshape((1, 3, 3, 1))
    for i in range(100):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    matrix = np.zeros((9, 4))
    matrix[[0, 1, 3, 4], 0] = kernel.reshape((-1, ))
    matrix[[1, 2, 4, 5], 1] = kernel.reshape((-1, ))
    matrix[[3, 4, 6, 7], 2] = kernel.reshape((-1, ))
    matrix[[4, 5, 7, 8], 3] = kernel.reshape((-1, ))

    _, s, _ = svd(matrix)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val_for_convolution(w, u, ip=1, padding='valid')[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-4


def test_singular_val_for_convolution():
    from numpy.linalg import svd
    w = K.placeholder([2, 2, 1, 1])
    u = K.placeholder([1, 3, 3, 1])

    f = K.function([w, u], max_singular_val_for_convolution(w, u, ip=100, padding='valid'))

    w_values = np.random.normal(size=[2, 2, 1, 1])
    u_values = np.random.normal(size=[1, 3, 3, 1])

    matrix = np.zeros((9, 4))

    matrix[[0, 1, 3, 4], 0] = w_values.reshape((-1, ))
    matrix[[1, 2, 4, 5], 1] = w_values.reshape((-1, ))
    matrix[[3, 4, 6, 7], 2] = w_values.reshape((-1, ))
    matrix[[4, 5, 7, 8], 3] = w_values.reshape((-1, ))

    _, s, _ = svd(matrix)
    sigma, u = f([w_values, u_values])

    assert np.abs(sigma - s[0]) < 1e-5


def test_sn_dense():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((5, ))
    out = SNDense(units=10, kernel_initializer=kernel_init, stateful=True)(inp)
    m = Model([inp], [out])
    x = np.arange(5 * 10).reshape((10, 5))
    for i in range(50):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_emb():
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)
    np.random.seed(0)
    inp = Input((1, ), dtype='int32')
    out = SNEmbeding(input_dim=10, output_dim=10, embeddings_initializer=kernel_init, stateful=True)(inp)
    m = Model([inp], [out])
    cls_val = (np.arange(5) % 3)[:,np.newaxis]
    for i in range(100):
        m.predict([cls_val])
    kernel = K.get_value(m.layers[1].embeddings)
    u_val = K.get_value(m.layers[1].u)

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])
    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_iterations():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((5, ))
    out = SNDense(units=10, kernel_initializer=kernel_init, spectral_iterations=50, stateful=True)(inp)
    m = Model([inp], [out])
    x = np.arange(5 * 10).reshape((10, 5))
    for i in range(1):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_conv2D():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    out = SNConv2D(kernel_size=(3, 3), padding='same', filters=10, kernel_initializer=kernel_init, stateful=True)(inp)
    m = Model([inp], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    for i in range(100):
        m.predict([x])

    kernel = K.get_value(m.layers[1].kernel)
    u_val = K.get_value(m.layers[1].u)

    kernel = kernel.reshape((-1, kernel.shape[3]))

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_conditional_conv():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    cls = Input((1, ), dtype='int32')
    out = SNConditionalConv11(number_of_classes=3, filters=10, kernel_initializer=kernel_init, stateful=True)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        m.predict([x, cls_val])

    kernel_all = K.get_value(m.layers[2].kernel)
    u_val_all = K.get_value(m.layers[2].u)

    for i in range(3):
        kernel = kernel_all[i]
        kernel = kernel.reshape((-1, kernel.shape[3]))
        u_val = u_val_all[i]

        _, s, _ = svd(kernel)

        w = K.placeholder(kernel.shape)
        u = K.placeholder(u_val.shape)
        max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

        assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_conditional_conv_with_renorm():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    cls = Input((1, ), dtype='int32')
    out = SNConditionalConv11(number_of_classes=3, filters=10, kernel_initializer=kernel_init, stateful=True, renormalize=True)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        m.predict([x, cls_val])

    kernel = K.get_value(m.layers[2].kernel)
    u_val = K.get_value(m.layers[2].u)

    kernel = kernel.reshape((-1, kernel.shape[-1]))
    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_conditional_dense():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input((4, ))
    cls = Input((1, ), dtype='int32')
    out = SNCondtionalDense(number_of_classes=3, units=10, kernel_initializer=kernel_init, stateful=True)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 4).reshape((5, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        m.predict([x, cls_val])

    kernel_all = K.get_value(m.layers[2].kernel)
    u_val_all = K.get_value(m.layers[2].u)

    for i in range(3):
        kernel = kernel_all[i]
        kernel = kernel.reshape((-1, kernel.shape[-1]))
        u_val = u_val_all[i]

        _, s, _ = svd(kernel)

        w = K.placeholder(kernel.shape)
        u = K.placeholder(u_val.shape)
        max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])
        assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_conditional_dense_with_renorm():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input((4, ))
    cls = Input((1, ), dtype='int32')
    out = SNCondtionalDense(number_of_classes=3, units=10, kernel_initializer=kernel_init, stateful=True, renormalize=True)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 4).reshape((5, 4))
    cls_val = (np.arange(5) % 3)[:,np.newaxis]

    for i in range(100):
        m.predict([x, cls_val])

    kernel = K.get_value(m.layers[2].kernel)
    u_val = K.get_value(m.layers[2].u)

    kernel = kernel.reshape((-1, kernel.shape[-1]))
    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_depthwise_with_renorm():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    cls = Input((1, ), dtype='int32')
    out = SNConditionalDepthwiseConv2D(number_of_classes=3, kernel_size=(3, 3), padding='same',
                                       filters=4, kernel_initializer=kernel_init, stateful=True)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    cls_val = np.zeros(shape=(5, 1))
    for i in range(100):
        m.predict([x, cls_val])

    kernel = K.get_value(m.layers[2].kernel)
    u_val = K.get_value(m.layers[2].u)

    kernel = kernel.reshape((-1, kernel.shape[3]))

    _, s, _ = svd(kernel)

    w = K.placeholder(kernel.shape)
    u = K.placeholder(u_val.shape)
    max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])

    assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_sn_depthwise():
    
    import numpy as np
    from numpy.linalg import svd
    def kernel_init(shape):
        return np.random.normal(size=shape)

    inp = Input((2, 3, 4))
    cls = Input((1, ), dtype='int32')
    out = SNConditionalDepthwiseConv2D(number_of_classes=3, kernel_size=(3, 3), padding='same', renormalize=False,
                                       filters=4, kernel_initializer=kernel_init, stateful=True)([inp, cls])
    m = Model([inp, cls], [out])
    x = np.arange(5 * 2 * 3 * 4).reshape((5, 2, 3, 4))
    cls_val = np.zeros(shape=(5, 1))
    for i in range(100):
        m.predict([x, cls_val])

    kernel_all = K.get_value(m.layers[2].kernel)
    u_val_all = K.get_value(m.layers[2].u)

    for cls in range(3):
        for c in range(4):
            kernel = kernel_all[cls, ..., c]
            kernel = kernel.reshape((-1, 1))
            u_val = u_val_all[4 * cls + c]

            _, s, _ = svd(kernel)

            w = K.placeholder(kernel.shape)
            u = K.placeholder(u_val.shape)
            max_sg_fun = K.function([w, u], [max_singular_val(w, u)[0]])
            assert np.abs(max_sg_fun([kernel, u_val]) - s[0])[0] < 1e-5


def test_conditional_dense():
    
    import numpy as np
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input((2,))
    cls = Input((1, ), dtype='int32')
    dence = ConditionalDense(number_of_classes=3, units=2, use_bias=True,
                            kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls])
    rs_inp = Reshape((1, 1, 2))([inp])
    cv_sep = ConditionalConv2D(number_of_classes=3, kernel_size=(1, 1), filters=2, padding='valid', use_bias=True,
                               kernel_initializer=kernel_init, bias_initializer=kernel_init)([rs_inp, cls])
    m = Model([inp, cls], [dence, cv_sep])
    x = np.arange(2 * 2).reshape((2, 2))
    cls = np.expand_dims(np.arange(2) % 3, axis=-1)
    out1, out2 = m.predict([x, cls])
    out2 = np.squeeze(out2, axis=(1, 2))

    assert np.sum(np.abs(out1 - out2)) < 1e-5


def test_conditional_conv11():
    
    import numpy as np
    def kernel_init(shape):
        np.random.seed(0)
        return np.random.normal(size=shape)

    inp = Input(batch_shape=(10, 10, 10, 10))
    cls = Input(batch_shape=(10, 1), dtype='int32')
    cv11 = ConditionalConv11(number_of_classes=3, filters=20,
                             kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls])
    cv_sep = ConditionalConv2D(number_of_classes=3, kernel_size=(1, 1), filters=20, padding='valid', use_bias=True,
                               kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls])
    m = Model([inp, cls], [cv11, cv_sep])
    x = np.arange(10 * 10 * 10 * 10).reshape((10, 10, 10, 10))
    cls = np.expand_dims(np.arange(10) % 3, axis=-1)
    out1, out2 = m.predict([x, cls])

    assert np.mean(np.abs(out1 - out2)) < 1e-2


def test_conditional_instance():
    
    import numpy as np
    def beta_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a
    inp = Input((2, 2, 1))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalInstanceNormalization(3, axis=-1, gamma_initializer=beta_init,
                                                           beta_initializer=beta_init)([inp, cls]))
    x = np.ones((3, 2, 2, 1))
    cls = np.expand_dims(np.arange(3), axis=-1)
    out = m.predict([x, cls])

    assert np.all(out[0] == 1)
    assert np.all(out[1] == 2)
    assert np.all(out[2] == 3)


def test_conditional_center_scale():
    
    import numpy as np
    def beta_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a
    inp = Input((2, 2, 1))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalCenterScale(3, axis=-1, gamma_initializer=beta_init,
                                                            beta_initializer=beta_init)([inp, cls]))
    x = np.ones((3, 2, 2, 1))
    cls = np.expand_dims(np.arange(3), axis=-1)
    out = m.predict([x, cls])

    assert np.all(out[0] == 2)
    assert np.all(out[1] == 4)
    assert np.all(out[2] == 6)


def test_center_scale():
    
    import numpy as np
    def beta_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a
    inp = Input((2, 2, 3))
    m = Model([inp], CenterScale(axis=-1, gamma_initializer=beta_init,
                                             beta_initializer=beta_init)(inp))
    x = np.ones((3, 2, 2, 3))
    out = m.predict(x)

    assert np.all(out[..., 0] == 2)
    assert np.all(out[..., 1] == 4)
    assert np.all(out[..., 2] == 6)


def test_conditional_bn():
    
    import numpy as np
    def beta_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a
    K.set_learning_phase(0)
    inp = Input((2, 2, 1))
    cls = Input((1, ), dtype='int32')
    out = ConditinalBatchNormalization(3, axis=-1, gamma_initializer=beta_init,
                                                   moving_variance_initializer=lambda sh: 0.666666666667 * np.ones(sh),
                                                   beta_initializer='zeros',
                                                   moving_mean_initializer=lambda sh: 2 * np.ones(sh))([inp, cls])
    m = Model([inp, cls], out)
    x = np.ones((3, 2, 2, 1))

    x[1] = x[1] * 2
    x[2] = x[2] * 3

    cls = np.expand_dims(np.arange(3), axis=-1)
    out = m.predict([x, cls])
    out = np.squeeze(out)

    assert np.all(np.abs(out[0] + 1.22) < 0.1)
    assert np.all(np.abs(out[1] - 0) < 0.1)
    assert np.all(np.abs(out[2] - 3.67) < 0.1)


def test_conditional_conv():
    
    import numpy as np
    def kernel_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a

    inp = Input((2, 2, 1))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalConv2D(number_of_classes=3, filters=1,
             kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls]))
    x = np.ones((3, 2, 2, 1))
    cls = np.expand_dims(np.arange(3), axis=-1)
    cls[2] = 0
    out = m.predict([x, cls])

    assert np.all(out[0] == 5)
    assert np.all(out[1] == 10)
    assert np.all(out[2] == 5)


def test_triangular_conv11():
    
    import numpy as np
    def kernel_init(shape):
        a = np.empty(shape)
        a[..., 0] = 1
        a[..., 1] = 2
        #a[1] = 2
        #a[2] = 3
        return a

    inp = Input((2, 2, 2))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalConv11(number_of_classes=1, filters=2, triangular=True,
                          kernel_initializer=kernel_init, bias_initializer='zeros')([inp, cls]))
    x = np.ones((1, 2, 2, 2))
    cls = np.expand_dims(np.arange(1), axis=-1)
    cls[:] = 0
    out = m.predict([x, cls])
    assert np.all(out[0, ..., 0] == 1)
    assert np.all(out[0, ..., 1] == 4)


def test_triangular_factorized_conv11():
    
    import numpy as np
    def kernel_init(shape):
        a = np.empty(shape)
        a[0] = 1
        if shape[0] != 1:
            a[1] = 2
        #a[1] = 2
        #a[2] = 3
        return a

    inp = Input((2, 2, 2))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], FactorizedConv11(number_of_classes=2, filters=2, filters_emb=1,
                                           kernel_initializer=kernel_init, bias_initializer='zeros')([inp, cls]))
    x = np.ones((2, 2, 2, 2))
    cls = np.expand_dims(np.arange(2), axis=-1)
    #cls[:] = 0
    out = m.predict([x, cls])
    #print np.squeeze(out[0])
    #print np.squeeze(out[1])
    assert np.all(out[0] == 2)
    assert np.all(out[1] == 2)


def test_deptwise_conv():
    
    import numpy as np
    def kernel_init(shape):
        a = np.empty(shape)
        a[0, ..., 0] = 1
        a[1, ..., 0] = 2
        a[2, ..., 0] = 3

        a[0, ..., 1] = 2
        a[1, ..., 1] = 3
        a[2, ..., 1] = 5

        return a

    inp = Input((2, 2, 2))
    cls = Input((1, ), dtype='int32')
    m = Model([inp, cls], ConditionalDepthwiseConv2D(number_of_classes=3, filters=2,
             kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init, bias_initializer=kernel_init)([inp, cls]))
    x = np.ones((3, 2, 2, 2))
    cls = np.expand_dims(np.arange(3), axis=-1)
    cls[2] = 0
    out = m.predict([x, cls])

    assert np.all(out[0, ..., 0] == 5)
    assert np.all(out[1, ..., 0] == 10)
    assert np.all(out[2, ..., 0] == 5)

    assert np.all(out[0, ..., 1] == 10)
    assert np.all(out[1, ..., 1] == 15)
    assert np.all(out[2, ..., 1] == 10)

def test_decorelation():
    
    import numpy as np
    def beta_init(shape):
        a = np.empty(shape)
        a[0] = 1
        a[1] = 2
        a[2] = 3
        return a
    K.set_learning_phase(1)
    inp = Input((10, 10, 64))
    decor_l = DecorelationNormalization(renorm=False)
    decor = decor_l(inp)
    decor_l.stateful = True
    out = decor

    m = Model([inp], [out])

    cov = 0.5 * np.eye(64) + 0.5 * np.ones((64, 64))

    x = np.random.multivariate_normal(mean=np.ones(64), cov=cov, size=(10, 10, 10))
    out = m.predict(x)

    out = np.reshape(out, [-1, out.shape[-1]])
    #print np.cov(out, rowvar=False)
    assert (np.mean(np.abs(np.cov(out, rowvar=False) - np.eye(64))) < 1e-3)

if __name__ == "__main__":
    test_conditional_conv()
    test_conditional_instance()
    test_conditional_conv11()
    test_triangular_conv11()
    test_conditional_dense()
    test_deptwise_conv()
    test_conditional_bn()
    test_decorelation()
    test_conditional_center_scale()
    test_center_scale()
    test_triangular_factorized_conv11()

    test_sn_emb()
    test_sn_conditional_dense_with_renorm()
    test_conv_with_conv_spectal()
    test_sn_conditional_dense()
    test_sn_conditional_conv()

    test_sn_conv2D()
    test_sn_dense()
    test_singular_val_for_convolution()
    test_conv_with_conv_spectal()
    test_iterations()
    test_sn_depthwise_with_renorm()
    test_sn_depthwise()
    test_sn_conditional_dense_with_renorm()
    test_sn_conditional_conv_with_renorm()
