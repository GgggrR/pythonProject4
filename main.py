import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import NullFormatter
from matplotlib.legend_handler import HandlerLine2D
from numpy.random import randn
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.animation import FuncAnimation

def line():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()


def line_point():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.show()


def styles():
    t = np.arange(0., 5., 0.2)
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


def multiple_figures():
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

    plt.subplot(212)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    plt.show()


def Annotating_text():
    ax = plt.subplot(111)

    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    line, = plt.plot(t, s, lw=2)

    plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    plt.ylim(-2, 2)
    plt.show()


def log_axes():
    np.random.seed(19680801)
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))
    plt.figure(1)

    # linear
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)

    # log
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)

    # symmetric log
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog')
    plt.title('symlog')
    plt.grid(True)

    # logit
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)

    plt.gca().yaxis.set_minor_formatter(NullFormatter())

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.show()


def other_types_of_plots():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)

    plt.title("sin cos")
    plt.plot(C)
    plt.plot(S)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
    plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")
    plt.show()


    plt.plot(X, C, )
    plt.plot(X, S, )
    plt.xlim(X.min() * 1.1, X.max() * 1.1)
    plt.ylim(C.min() * 1.1, C.max() * 1.1)
    plt.show()


    plt.plot(X, C, )
    plt.plot(X, S, )
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks([-1, 0, +1],
               [r'$-1$', r'$0$', r'$+1$'])
    plt.show()

    plt.plot(X, C, )
    plt.plot(X, S, )
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.show()


    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
    plt.plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine")
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks([-1, 0, +1],
               [r'$-1$', r'$0$', r'$+1$'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.legend(loc='upper left', frameon=False)
    plt.show()


    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
    plt.plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine")
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks([-1, 0, +1],
               [r'$-1$', r'$0$', r'$+1$'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.legend(loc='upper left', frameon=False)

    t = 2 * np.pi / 3
    plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=1.5, linestyle="--")
    plt.scatter([t, ], [np.cos(t), ], 50, color='blue')

    plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
                 xy=(t, np.sin(t)), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.plot([t, t], [0, np.sin(t)], color='red', linewidth=1.5, linestyle="--")
    plt.scatter([t, ], [np.sin(t), ], 50, color='red')

    plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
                 xy=(t, np.cos(t)), xycoords='data',
                 xytext=(-90, -50), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()


def using_text():
    fig = plt.figure()
    fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('axes title')

    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

    ax.text(3, 8, 'boxed italics text in data coords', style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

    ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')

    ax.text(0.95, 0.01, 'colored text in axes coords',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=15)

    ax.plot([2], [1], 'o')
    ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.axis([0, 10, 0, 10])

    plt.show()


def using_mathtext():
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)

    plt.plot(t, s)
    plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
    plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
    plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
             fontsize=20)
    plt.xlabel('time (s)')
    plt.ylabel('volts (mV)')
    plt.show()


def using_legends():
    plt.subplot(211)
    plt.plot([1, 2, 3], label="test1")
    plt.plot([3, 2, 1], label="test2")
    # Place a legend above this subplot, expanding itself to
    # fully use the given bounding box.
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(223)
    plt.plot([1, 2, 3], label="test1")
    plt.plot([3, 2, 1], label="test2")
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    line1, = plt.plot([1, 2, 3], label="Line 1", linestyle='--')
    line2, = plt.plot([3, 2, 1], label="Line 2", linewidth=4)

    # Create a legend for the first line.
    first_legend = plt.legend(handles=[line1], loc=1)

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

    # Create another legend for the second line.
    plt.legend(handles=[line2], loc=4)
    plt.show()

    line1, = plt.plot([3, 2, 1], marker='o', label='Line 1')
    line2, = plt.plot([1, 2, 3], marker='o', label='Line 2')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()

    z = randn(10)

    red_dot, = plt.plot(z, "ro", markersize=15)
    # Put a white cross over some of the data.
    white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)

    plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
    plt.show()

    class AnyObject(object):
        pass

    class AnyObjectHandler(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                       edgecolor='black', hatch='xx', lw=3,
                                       transform=handlebox.get_transform())
            handlebox.add_artist(patch)
            return patch

    plt.legend([AnyObject()], ['My first handler'],
               handler_map={AnyObject: AnyObjectHandler()})
    plt.show()

    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Ellipse(xy=center, width=width + xdescent,
                                 height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="green",
                        edgecolor="red", linewidth=3)
    plt.gca().add_patch(c)
    plt.legend([c], ["An ellipse, not a rectangle"],
               handler_map={mpatches.Circle: HandlerEllipse()})
    plt.show()



fig = plt.figure(figsize=(6,6), facecolor='white')
ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)
n = 50
size_min = 50
size_max = 50*50

P = np.random.uniform(0,1,(n,2))

C = np.ones((n,4)) * (0,0,0,1)
C[:,3] = np.linspace(0,1,n)


S = np.linspace(size_min, size_max, n)

scat = ax.scatter(P[:,0], P[:,1], s=S, lw = 0.5,
                  edgecolors = C, facecolors='None')


ax.set_xlim(0,1), ax.set_xticks([])
ax.set_ylim(0,1), ax.set_yticks([])

def update(frame):
    global P, C, S

    # Каждое кольцо делается более прозрачным
    C[:,3] = np.maximum(0, C[:,3] - 1.0/n)

    # Каждое кольцо увеличивается
    S += (size_max - size_min) / n

    # Сбросить кольцо
    i = frame % 50
    P[i] = np.random.uniform(0,1,2)
    S[i] = size_min
    C[i,3] = 1

    # Обновить scatter object
    scat.set_edgecolors(C)
    scat.set_sizes(S)
    scat.set_offsets(P)

    return scat,

animation = FuncAnimation(fig, update, interval=10, blit=True, frames=200)




line()
line_point()
styles()
multiple_figures()
Annotating_text()
other_types_of_plots()
plt.show()
log_axes()
using_text()
using_mathtext()
using_legends()
