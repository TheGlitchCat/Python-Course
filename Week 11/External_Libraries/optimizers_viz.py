#@title Optim_viz Library
############
### Adaptado de wassname/viz_torch_optim para compatibilidad con Pytorch  1.1 y modularización.
############

import torch
import numpy as np
from torch import optim
from torch.optim import Optimizer
from torch.optim import lr_scheduler
import numpy
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from collections import defaultdict
from itertools import zip_longest
from functools import partial
from matplotlib import rcParams
import datetime

torch.set_default_tensor_type('torch.DoubleTensor')
dtype=np.float64
rcParams['figure.figsize']=(10,10)


class Problem(object):
    def __init__(self, f, df, minima, x0, bounds=[[-5,5],[-5,5]], lr=1e-3, steps=3000, noise=dict(m=0,c=0)):
        """
        Problem setup
        
        Params:
        - f: function [x1,x2] => z
        - df: derivative function ([x1,x2]=>[dx1,dx2])
        - minima: where the function has a minima
        - self: bounds
        
        - x0: suggested start
        - lr: suggested learning rate
        - steps: suggested steps
        """
        def f_noise(t):
            """Add some noise"""
            t = to_tensor(t)
            c = torch.rand(t[0].size()) * noise['c']
            m = 1 + torch.rand(t[0].size()) * noise['m']
            z = f(t)
            return  m * z + c 
        self.f = f_noise
        self._f = f
        self.df = df
        self.x0 = x0
        self.bounds = bounds
        self.minima = minima
        self.lr = lr
        self.steps = steps
        
        self.xmin = bounds[0][0]
        self.xmax = bounds[0][1]
        self.ymin = bounds[1][0]
        self.ymax = bounds[1][1]
        
        
"""Valley"""


def beales(tensor):
    """Beales function, like a valley"""
    x, y = tensor
    x = to_tensor(x)
    y = to_tensor(y)
    # + noise(x,y)
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
  
def dbeales(tensor):
    x, y = tensor
    x = to_tensor(x)
    y = to_tensor(y)
    dx = 2 * (x * y**3 - x + 2.625) * (y**3 - 1) + 2 * (x * y**2 -
                                                        x + 2.25) * (y**2 - 1) + 2 * (x * y - x + 1.5) * (y - 1)
    dy = 6 * (x * y**3 - x + 2.625) * x * y**2 + 4 * \
        (x * y**2 - x + 2.25) * x * y + 2 * (x * y - x + 1.5) * x
    return torch.stack([dx, dy], 1)[0]
  
def to_tensor(x):
    # TODO: I'm sure there's a proper way to do this
    if isinstance(x, np.ndarray):
        return torch.Tensor(x.astype(dtype))
    if isinstance(x, list):
        return torch.Tensor(x)
    elif isinstance(x, (float, int, numpy.generic)):
        return torch.Tensor([float(x)])
    else:
        return x


def test_f(f, df, constructor, steps=150, x0=[-4,-1], solution=[-2,0], scheduler=None, exact=False):
    """
    modified from https://github.com/pytorch/pytorch/blob/master/test/test_optim.py

    params:
    scheduler: e.g. scheduler = torch.optim.CyclicLR(optimizer)
    
    """
    state = {}

    # start
    params = torch.tensor(x0, requires_grad=True)
    optimizer = constructor([params])
    initial_lr = optimizer.param_groups[0]['lr']
    if scheduler:
        _scheduler = scheduler(optimizer, initial_lr)

    solution = torch.Tensor(solution)
    initial_dist = params.data.dist(solution)

    def eval():
        optimizer.zero_grad()
        loss = f(params)
        loss.backward()
        return loss

    data=[]
    dist=[]
    lrs=[]
    for i in range(steps):
        
        loss = optimizer.step(eval)
        if scheduler:
            _scheduler.batch_step()
        
        # record
        dist.append(loss.squeeze().data.numpy()) # loss
        data.append(params.data.numpy().copy())
        lrs.append(optimizer.param_groups[0]['lr'])
    return np.array(data), np.array(dist), lrs


class CyclicLR(object):
    """
    from https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py
    not merged into pytorch yet https://github.com/pytorch/pytorch/pull/2016

    Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

def build_beales_problem():
    return Problem(
        f=beales,
        df=dbeales,
        minima=np.array([3., 0.5]),
        bounds=[[-4.5,4.5],[-4.5,4.5]],
        x0=[2,1.7],
        steps=3400,
        lr=3e-3,
        noise=dict(m=0.13, c=7),
    )

def build_optimizers(lr):   
    return dict(
        # Good high dimensional optimizers sometimes do poorly in low D spaces, so we will lower the LR on simple optimisers
        # need smaller lr's sometimes
        SGD= lambda params: optim.SGD(params, lr=lr/80),
        momentum = lambda params: optim.SGD(params, lr=lr/80, momentum=0.9, nesterov=False, dampening=0),
        momentum_dampen = lambda params: optim.SGD(params, lr=lr/80, momentum=0.9, nesterov=False, dampening=0.3),
        nesterov = lambda params: optim.SGD(params, lr=lr/80, momentum=0.9, nesterov=True, dampening=0),
        nesterov_decay = lambda params: optim.SGD(params, lr=lr/80, momentum=0.9, nesterov=True, weight_decay=1e-4, dampening=0),
        
        # need larger lr's sometimes
        Adadelta = lambda params: optim.Adadelta(params),
        Adagrad = lambda params: optim.Adagrad(params, lr=lr*20),
        
        # 
        Adamax = lambda params: optim.Adamax(params, lr=lr*20),
        RMSprop = lambda params: optim.RMSprop(params, lr=lr*10),
        Adam = lambda params: optim.Adam(params, lr=lr*10),
    #     Adam_decay = lambda params:  optim.Adam(params, lr=lr*10, weight_decay=1e-9),
        
        # need to read about these, might not be comparable
    #     ASGD = lambda params: optim.ASGD(params, lr=lr),
    #     Rprop = lambda params: optim.Rprop(params, lr=lr),
    #     LBFGS = lambda params: optim.LBFGS(params),
    )

def build_params(problem):
    xmin = problem.xmin
    xmax = problem.xmax
    ymin = problem.ymin
    ymax = problem.ymax
    ystep = xstep= (xmax-xmin)/200.0
    zeps = 1.1e-0 # we don't want the minima to be actual zero or we wont get any lines shown on a log scale
    z_min = problem.f(problem.minima).data.numpy().reshape(1)
    minima_ = problem.minima.reshape(-1, 1)
    _x0 = np.array([problem.x0]).T
    # and x, y, z
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = problem.f([x, y]).data.numpy() 
    if z.min()<z_min[0]:
      print('WARN: your minima is not the true minima', z_min[0], z.min())
      z_min[0]=z.min()  
    z += -z_min[0] + zeps  # we shift everything up so the min is 1, so we can show on log scale
    logzmax=np.log(z.max()-z.min()+zeps)            
    return dict(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        ystep=ystep,
        zeps=zeps,
        z_min=z_min,
        minima_=minima_,
        _x0=_x0,
        x=x,
        y=y,
        z=z,
        logzmax=logzmax  
    )



def plot_minimized_function(params):
    x = params["x"]
    y = params["y"]
    z = params["z"]
    logzmax = params["logzmax"]
    minima = params["minima_"]
    x0 = params['_x0']
    ax = plt.gca()
    cm=ax.contour(x, y, z, levels=np.logspace(0, logzmax//2, 55), norm=LogNorm(), cmap=plt.cm.jet, alpha=0.15)
    plt.colorbar(cm)
    ax.plot(*minima, 'r*', markersize=10)
    ax.plot(*x0, 'r+', markersize=10)
    plt.title('debug: grid')

    plt.show()





def save_optim_animation():
    ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    rcParams['figure.figsize']=(10,10)
    rcParams['figure.dpi']=100
    rcParams['animation.writer']='ffmpeg' # faster than fmmpeg
    rcParams['savefig.dpi']=180
    rcParams['animation.codec']='h264'

    # rcParams['savefig.bbox'] = 'tight'
    seconds = 20
    fps = 30 # too low and you will miss the fast moving ones
    rcParams['animation.bitrate']=-1 #fps*1000//25 # ~1mb/s
    cuttoff=problem.steps//1 # if we want to crop the data to X steps, judging by the loss plot

    decimation = int(np.round(cuttoff/(seconds*fps))) or 1 # don't need to plot every step
    # assert cuttoff>seconds*fps
    decimation, cuttoff,seconds*fps, problem.steps
    if scheduler:
        params = torch.tensor([0.0,1.0], requires_grad=True)
        optimizer = optim.Adam([params])
        scheduler_name = type(scheduler(optimizer, 0.1)).__name__
    else:
        scheduler_name='None'

    title=f'function: f{problem._f.__name__}'
    if scheduler:
        title += f', scheduler: {scheduler_name}'
    save_file = course_path + 'videos/{name:}_{scheduler}_{ts:}'.format(name=problem._f.__name__, ts=ts, scheduler=scheduler_name)

    # from http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    class TrajectoryAnimation(animation.FuncAnimation):
        
        def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None, 
                     interval=60, repeat_delay=5, blit=True, **kwargs):

            if fig is None:
                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = ax.get_figure()
            else:
                if ax is None:
                    ax = fig.gca()

            self.fig = fig
            self.ax = ax
            
            self.paths = paths

            if frames is None:
                frames = max(path.shape[1] for path in paths)
      
            self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                          for _, label in zip_longest(paths, labels)]
            self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
                           for line in self.lines]

            super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                      frames=frames, interval=interval, blit=blit,
                                                      repeat_delay=repeat_delay, **kwargs)

        def init_anim(self):
            for line, point in zip(self.lines, self.points):
                line.set_data([], [])
                point.set_data([], [])
            return self.lines + self.points

        def animate(self, i):
            for line, point, path in zip(self.lines, self.points, self.paths):
                line.set_data(*path[::,:i])
                point.set_data(*path[::,i-1:i])
            return self.lines + self.points

    fig, ax = plt.subplots(figsize=(12, 8))
    # fig.set_tight_layout(True)
    ax.contour(x, y, z, levels=np.logspace(0, logzmax//2, 35), norm=LogNorm(), cmap=plt.cm.jet, alpha=0.5)
    ax.plot(*minima_, 'r*', markersize=10)
    ax.plot(*problem.x0, 'r+', markersize=10)

    ax.set_title('{} (github.com/wassname/viz_torch_optim)'.format(title))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    anim = TrajectoryAnimation(*paths[:,:,:cuttoff:decimation], labels=methods, ax=ax, interval=1000//fps)

    ax.legend(loc='upper left')

    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))

    import matplotlib.animation
    import logging
    _log = logging.getLogger(matplotlib.animation.__file__)
    _log.setLevel(logging.DEBUG)

    anim.save(save_file+'_2d'+'.mp4', fps=fps, writer="ffmpeg", codec='h264')
    #anim.save(save_file+'2d'+'.gif', fps=fps, codec='gif', writer="imagemagick")

def run_optimizers(problem, constructors, params):
    lr = problem.lr
    xmin = params["xmin"]
    xmax = params["xmax"]
    ymin = params["ymin"]
    ymax = params["ymax"]
    ystep = params["ystep"]
    zeps = params["zeps"]
    z_min = params["z_min"]
    minima_ = params["minima_"]
    _x0 = params["_x0"]
    x = params["x"]
    y = params["y"]
    z = params["z"]
    logzmax = params["logzmax"]

    results = {}
    distance = {}
    scheduler = lambda optimizer, initial_lr: CyclicLR(optimizer, base_lr=1e-5, max_lr=initial_lr, step_size=100, mode='exp_range', gamma=0.9983)
    for name, constructor in tqdm(constructors.items()):
        data, dist, lrs = test_f(problem.f, problem.df, constructor, x0=problem.x0, steps=problem.steps, scheduler=scheduler)
        results[name] = data
        distance[name] = dist
    methods = constructors.keys()
    paths = np.array([path.T for path in results.values()]) # should be (2,N) each
    zpaths = np.array([distance[name] - z_min[0] + zeps for name in methods])
    
    # Log z's
    for i, name in enumerate(results):
        zmax = zpaths[i][np.isfinite(zpaths[i])].max()
        print(name, zmax, '\t', np.isfinite(zmax).all(), '\t', zmax.max()>z[:,0].max())     
    # clip zpaths
    zmax = z.max()
    zpaths[np.isfinite(zpaths)==False]=zmax
    zpaths = np.clip(zpaths, 0, zmax)
    return zpaths, results, distance, lrs

def plot_optim_journeys(zpaths, results, distance, lrs, params):
    xmin = params["xmin"]
    xmax = params["xmax"]
    ymin = params["ymin"]
    ymax = params["ymax"]
    ystep = params["ystep"]
    zeps = params["zeps"]
    z_min = params["z_min"]
    minima_ = params["minima_"]
    _x0 = params["_x0"]
    x = params["x"]
    y = params["y"]
    z = params["z"]
    logzmax = params["logzmax"]
    
    # Preview plots
    # static preview 2d to let you debug your steps and learning rate

    # loss
    for i, name in enumerate(results):
        plt.plot(np.abs(distance[name]), label=name)
    plt.legend()
    plt.title('loss (mae)')
    plt.ylim(0,zpaths[:,0].mean()*1.3)
    plt.show()


    # Position
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    for name in results:
        plt.scatter(*results[name].T, label=name, s=1)
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)

    cm=ax.contour(x, y, z, levels=np.logspace(0, logzmax//2, 35), norm=LogNorm(), cmap=plt.cm.jet, alpha=0.15)
    # plt.colorbar(cm)
    ax.plot(*minima_, 'r*', markersize=10)
    ax.plot(*_x0, 'r+', markersize=10)
    plt.title('debug: paths')

    plt.show()

    # lr
    plt.plot(lrs)
    # plt.yscale('log')
    plt.title('learning rate')
    plt.show()

def plot_3D_journey(zpaths, results, problem, params):
    xmin = params["xmin"]
    xmax = params["xmax"]
    ymin = params["ymin"]
    ymax = params["ymax"]
    ystep = params["ystep"]
    zeps = params["zeps"]
    z_min = params["z_min"]
    minima_ = params["minima_"]
    _x0 = params["_x0"]
    x = params["x"]
    y = params["y"]
    z = params["z"]
    logzmax = params["logzmax"]
    # static preview 3d
    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50,azim=-95)

    ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.25, cmap=plt.cm.jet)
    ax.plot(*minima_, problem.f(minima_).data.numpy(), 'r*', markersize=10)
    ax.plot(*_x0, problem.f(_x0).data.numpy(), 'r+', markersize=10)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    # anim = TrajectoryAnimation3D(*paths, zpaths=zpaths, labels=methods, ax=ax)
    # quick plot to let you debug your steps and learning rate
    ax = plt.gca()
    for i, name in enumerate(results):
        ax.scatter3D(*results[name].T, zpaths[i], label=name, s=1)
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)

    ax.legend(loc='upper right')