"""
DOCSTRING
"""
import matplotlib.pyplot as pyplot
import numpy
import os
import PIL.Image
import tensorflow

data_dir = 'model'
img_noise = numpy.random.uniform(size=(224,224,3)) + 100.0
model_fn = 'tensorflow_inception_graph.pb'
graph = tensorflow.Graph()
sess = tensorflow.InteractiveSession(graph=graph)

with tensorflow.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
    graph_def = tensorflow.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tensorflow.placeholder(numpy.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tensorflow.expand_dims(t_input-imagenet_mean, 0)
tensorflow.import_graph_def(graph_def, {'input':t_preprocessed})
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

def calc_grad_tiled(img, t_grad, tile_size=512):
    """
    Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over multiple iterations.
    """
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = numpy.random.randint(sz, size=2)
    img_shift = numpy.roll(numpy.roll(img, sx, 1), sy, 0)
    grad = numpy.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return numpy.roll(numpy.roll(grad, -sx, 1), -sy, 0)

def rename_nodes(graph_def, rename_func):
    """
    DOCSTRING
    """
    res_def = tensorflow.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def

def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    """
    DOCSTRING
    """
    t_score = tensorflow.reduce_mean(t_obj) 
    t_grad = tensorflow.gradients(t_score, t_input)[0]
    img = img0
    octaves = []
    for _ in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, numpy.int32(numpy.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (numpy.abs(g).mean()+1e-7))
        showarray(img/255.0)

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    """
    DOCSTRING
    """
    t_score = tensorflow.reduce_mean(t_obj)
    t_grad = tensorflow.gradients(t_score, t_input)[0]
    img = img0.copy()
    for _ in range(iter_n):
        g, _ = sess.run([t_grad, t_score], {t_input:img})
        g /= g.std()+1e-8
        img += g*step
    showarray(visstd(img))

def resize(img, size):
    """
    DOCSTRING
    """
    img = tensorflow.expand_dims(img, 0)
    return tensorflow.image.resize_bilinear(img, size)[0,:,:,:]

resize = tffunc(numpy.float32, numpy.int32)(resize)

def showarray(a):
    """
    DOCSTRING
    """
    a = numpy.uint8(numpy.clip(a, 0, 1)*255)
    pyplot.imshow(a)
    pyplot.show()

def strip_consts(graph_def, max_const_size=32):
    """
    Strip large constant values from graph_def.
    """
    strip_def = tensorflow.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def T(layer):
    """
    Helper for getting layer output tensor
    """
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    """
    Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    """
    placeholders = list(map(tensorflow.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def visstd(a, s=0.1):
    """
    Normalize the image range for visualization
    """
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
img0 = PIL.Image.open('data/input.jpg')
img0 = numpy.float32(img0)
render_deepdream(tensorflow.square(T('mixed4c')), img0)
