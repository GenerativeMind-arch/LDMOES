

''' Calculates the Frechet Inception Distance (FID).

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a model.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function  #division：让 / 操作符进行真正的除法（例如，5 / 2 会得到 2.5 而不是 2）

from PIL import Image
import imageio
import numpy as np
import os
import gzip, pickle  #用于序列化和反序列化 Python 对象，将对象保存到文件或从文件加载对象
import tensorflow as tf  #是一个广泛使用的机器学习和深度学习框架。它用于构建、训练和评估神经网络模型。通过 tf 来简化调用
from imageio import imread
#scipy是python的一个科学计算库，包含线性代数、积分、傅里叶变换等计算，linalg就是线性代数模块
from scipy import linalg
#是文件和文件路径管理库
import pathlib
#urllib 是 Python 的网络请求库，用于下载、读取和解析 URL 资源
import urllib
#warnings 是 Python 的标准库，用于发出警告信息。可以在代码中使用它来发出警告，提醒开发者某些行为可能不推荐或会产生问题，但不会中断程序的执行。
import warnings

class InvalidFIDException(Exception):

    pass

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile( pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString( f.read())
        #_ 表示返回值被丢弃，因为我们并不关心返回的操作句柄。实际图的导入过程会在 TensorFlow 内部完成。
        _ = tf.import_graph_def( graph_def, name='')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
#上述注释表明这段代码是用来处理 Inception 网络的，且来源于 OpenAI 提供的改进版 GAN 代码库中的 inception_score 模块。
#sess是tensorflow中的一个会话对象，用于执行计算并返回结果
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    #pool_3 是 Inception 网络中的一个池化层（通常是池化操作的输出层）
    #0应该是代表池化层的输出
    layername = 'pool_3:0'
    #获取pool3层的输出张量
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
              #shape = [s.value for s in shape] TF 1.x
              #shape 是一个包含维度信息的对象（如 [batch_size, height, width, channels]）
              shape = [s for s in shape] #TF 2.x
              new_shape = []
              for j, s in enumerate(shape):
                #shape的形状就是TensorFlow对象，应该就是b,h,w,c
                #j表示是不是第一个维度，s表示第一个维度的值是不是1
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3

#定义一个计算池化层 pool_3 激活值的函数   verbose源代码中=False
def get_activations(images, sess, batch_size=1000, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    #计算图像数据的总数量
    n_images = images.shape[0]
    #批量大小不能大于图片总数量，否则会报错，并将批量大小设置为图片总数量
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    #打印batch_size大小
    print(batch_size)
    #计算总batch数
    n_batches = n_images//batch_size # drops the last batch if < batch_size，即如果最后一个batch的大小小于batch_size，就将最后一个batch的数据去除
    #初始化一个空numpy数组，组的大小是 (n_batches * batch_size, 2048)，即存储每个批次的激活值，每个图像对应一个长度为 2048 的激活向量。
    pred_arr = np.empty((n_batches * batch_size,2048))
    for i in range(n_batches):
        if verbose:
                                                                #end="" 表示不换行，flush=True 确保输出立即显示
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
        
        batch = images[start:end]
        pred = sess.run(inception_layer, {'ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch.shape[0],-1)
    if verbose:
        print(" done")
    return pred_arr


"""Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """
#计算 Frechet Distance(FID)的函数
#eps是一个非常小的值（用来避免数值不稳定的情况）
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    #下面两行代码是在确保 mu1 和 mu2 都是至少一维的。如果输入是标量，则会将其转换为一维数组。这是为了保证之后对均值进行一致的操作
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    #确保 sigma1 和 sigma2 都是至少二维的协方差矩阵。即便是 1x1 的矩阵，也会被转换为二维数组。
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    #检查确保两个分布的均值和协方差矩阵的形状都是相同的
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    #两个分布的均值的差
    diff = mu1 - mu2

    # product might be almost singular
    #计算（根号下两个分布的协方差矩阵点积)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    #如果 covmean 包含无穷大（inf）或 NaN，说明计算的协方差矩阵平方根是奇异的，可能导致数值不稳定。此时，offset 被添加到协方差矩阵的对角线上，以避免奇异性。
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component;由于数值误差，协方差矩阵的平方根可能会产生非常小的虚部。这里检查 covmean 是否为复数矩阵，并确保其虚部几乎为零。
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    #计算两个矩阵点积的迹
    tr_covmean = np.trace(covmean)

    #FID的计算公式：
    #return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return diff.dot(diff) + np.trace(sigma1+sigma2-2*covmean)

"""Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
#定义一个计算 Frechet Inception Distance (FID) 所需要的统计量（均值 mu 和协方差矩阵 sigma）的过程的函数
#其中， verbose源代码为False
def calculate_activation_statistics(images, sess, batch_size=1000, verbose=True):
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    #rowvar=False 表示按列计算协方差（每个特征维度之间的协方差）。这相当于计算样本之间的协方差矩阵。
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    

#------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements.txt, at the cost of slightly reduced efficiency.
# - Pyrestone
#------------------

"""Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
#该函数用于批量加载图像并返回它们的像素值
# def load_image_batch(files):
#     return np.array([imageio.imread(str(fn)).astype(np.float32) for fn in files])
def load_image_batch(files):
    images = []
    for fn in files:
        image = np.array(Image.open(str(fn))).astype(np.float32)
        #image = image / 127.5 - 1.0  # 关键修正：归一化到[-1,1]
        images.append(image)
    return np.array(images)


"""Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
#定义了一个计算文件中图像的激活值的函数
#verbose的源代码为False
def get_activations_from_files(files, sess, batch_size=1000, verbose=True):
    inception_layer = _get_inception_layer(sess)
    n_imgs = len(files)
    if batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs
    n_batches = n_imgs//batch_size + 1
    pred_arr = np.empty((n_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_imgs:
            end = start+batch_size
        else:
            end = n_imgs
        
        batch = load_image_batch(files[start:end])
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
        del batch #clean up memory
    if verbose:
        print(" done")
    return pred_arr

def calculate_activation_statistics_from_files(files, sess, batch_size=1000, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_files(files, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)

#该函数根据传入的路径（path）来判断文件类型，并分别处理不同类型的数据；所以我们在计算FID指标时，也不一定非要使用npz格式
def _handle_path(path, sess, low_profile=False):
    #检查传入的路径 path 是否以 .npz 结尾，即判断路径是否指向一个 NumPy 存档文件；
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        #使用 pathlib.Path.glob 方法查找路径下的所有 .jpg 和 .png 文件，并将它们保存在 files 列表中。这里使用了通配符 *.jpg 和 *.png 来匹配所有图片文件。
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        if low_profile:
            m, s = calculate_activation_statistics_from_files(files, sess)
        else:
            #x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
            x = np.array([np.array(Image.open(str(fn))).astype(np.float32) for fn in files])
            #x = x / 127.5 - 1.0
            m, s = calculate_activation_statistics(x, sess)
            del x #clean up memory
    return m, s

#定义了一个名为calculate_fid_given_paths的函数，它接受三个参数;
#paths：一个包含两个路径的列表，路径指向两组图片（分别是生成的图片和真实的图片）
def calculate_fid_given_paths(paths, inception_path, low_profile=False):
    ''' Calculates the FID of two paths. '''
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)
    #调用create_inception_graph函数来创建Inception模型的计算图，inception_path提供模型路径。create_inception_graph会将该模型加载到图中，准备好进行图像特征提取。
    create_inception_graph(str(inception_path))

    with tf.compat.v1.Session() as sess:
        #初始化所有变量。这是TensorFlow 1.x中必需的步骤，确保图中所有的变量都被初始化。
        sess.run(tf.compat.v1.global_variables_initializer())
        #计算两个分布的均值和协方差矩阵
        m1, s1 = _handle_path(paths[0], sess, low_profile=low_profile)
        m2, s2 = _handle_path(paths[1], sess, low_profile=low_profile)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    #该参数是生成的图像文件或 .npz 统计文件的路径；要求必须是两部分文件
    parser.add_argument("--path", type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files')
    #添加一个名为--inception（或-i）的命令行参数，类型为str，默认值为None。该参数用于指定Inception模型的路径。如果不提供路径，程序会尝试下载该模型
    parser.add_argument("-i", "--inception", type=str, default=None,
        help='Path to Inception model (will be downloaded if not provided)')
    #添加一个名为--gpu的命令行参数，类型为str，默认值为空字符串""。该参数允许用户指定要使用的GPU设备。如果不提供，默认使用CPU。
    parser.add_argument("--gpu", default="", type=str,
        help='GPU to use (leave blank for CPU only)')
    #添加一个可选参数 --lowprofile，该参数是一个布尔标志（store_true）。如果用户提供此参数，则会启用低内存模式，仅将一批图像保存在内存中，从而减少内存占用，但可能会稍微降低速度。
    parser.add_argument("--lowprofile", action="store_true",
        help='Keep only one batch of images in memory at a time. This reduces memory footprint, but may decrease speed slightly.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    fid_value = calculate_fid_given_paths(args.path, args.inception, low_profile=args.lowprofile)
    print("FID: ", fid_value)