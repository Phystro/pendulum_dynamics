import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as patches

# assign constants
g: float = 9.81
m: float = 1.0
l: float = 1.0

# Initial conditions: theta, velocity
theta0: float = np.deg2rad(30)
theta_dot0: float = 0.0

# system of differential equations
# y(theta, theta_dot) = statevector y[0] is theta, y[1] is theta_dot
# returns: y_dot(theta_dot, theta_dot_dot)
def pendulum_ODE(t: float, y: np.ndarray) -> np.ndarray:
    theta: float = y[0]
    theta_dot: float = y[1]
    theta_dot_dot: float = (-g/l) * np.sin(theta)
    return np.array([theta_dot, theta_dot_dot])

# solve the ODe, 30 fps
sol = solve_ivp(fun=pendulum_ODE, t_span=[0, 5], y0=[theta0, theta_dot0], t_eval=np.linspace(0, 5, 30*5))
print(sol)
print(type(sol))
print(type(sol.t))
print(type(sol.y[0]))

# output of solver
theta = sol.y[0]
theta_dot = sol.y[1]
t = sol.t

# convert from radians to degrees
theta_deg = np.rad2deg(theta)
theta_dot_deg = np.rad2deg(theta_dot)

def static_pqvt_plot():
    # plot of theta and theta_dot vs time
    plt.plot(t, theta_deg, 'r', linewidth=2, label=r'$\theta$')
    plt.plot(t, theta_dot_deg, 'b', linewidth=2, label=r'$\dot \theta$')
    plt.legend()
    plt.title('Simple Pendulum')
    plt.xlabel('time (s)')
    plt.ylabel(r'$\theta (^\circ)$, $\dot \theta (^\circ/s)$')
    plt.grid()
    plt.show()

def animated_pqvt_plot():
    # Animation of theta, theta_dot vs time
    fig, axis = plt.subplots(nrows=1, ncols=1)
    theta_curve, = axis.plot(t[0], theta_deg[0], 'r')
    theta_dot_curve, = axis.plot(t[0], theta_dot_deg[0], 'b')

    axis.set_title('Simple Pendulum: Angular position, Velocity vs Time')
    axis.set_xlim(0, 5)
    axis.set_ylim(-100, 100)
    axis.set_ylabel(r'$\theta (^\circ)$, $\dot \theta (^\circ/s)$')
    axis.set_xlabel('Time (s)')
    axis.legend([r'$\theta$', r'$\dot \theta$'])
    axis.grid()

    def animate(frame: int):
        theta_curve.set_data(t[:frame+1], theta_deg[:frame+1])
        theta_dot_curve.set_data(t[:frame+1], theta_dot_deg[:frame+1])
        return theta_curve, theta_dot_curve,

    # save video @ 30 fps
    anim = animation.FuncAnimation(fig, animate, frames=len(t), blit=True)
    ffmpeg_writer = animation.FFMpegWriter(fps=30)
    anim.save('time_domain.mp4', writer=ffmpeg_writer)

def static_phase_space_plot():
    # plot of phase diagram of theta, theta_dot
    plt.plot(theta_deg, theta_dot_deg, 'b')
    plt.title('Simple Pendulum: Phase Diagram')
    plt.xlabel(r'$\theta (^\circ)$')
    plt.ylabel(r'$\dot \theta (^\circ/s)$')
    plt.grid()
    plt.show()

def animated_phase_space_plot():
    # animated phase diagram plot of theta, theta_dot
    fig, axis = plt.subplots()
    phase_curve, = axis.plot(theta_deg[0], theta_dot_deg[0], 'b')
    phase_dot, = axis.plot(theta_deg[0], theta_dot_deg[0], 'ro')

    axis.set_title('Simple Pendulum: Phase Diagram')
    axis.set_xlim(-35, 35)
    axis.set_ylim(-100, 100)
    axis.set_xlabel(r'$\theta (^\circ)$')
    axis.set_ylabel(r'$\dot \theta (^\circ/s)$')
    axis.grid()

    def animate(frame: int):
        phase_curve.set_data(theta_deg[:frame+1], theta_dot_deg[:frame+1])
        phase_dot.set_data(theta_deg[frame:frame+1], theta_dot_deg[frame:frame+1])
        return phase_curve, phase_dot,

    anim = animation.FuncAnimation(fig, animate, frames=len(t), blit=True)
    ffmpeg_writer = animation.FFMpegWriter(fps=30)
    anim.save('phase_space_diagram.mp4', writer=ffmpeg_writer)


# Create animation of pendulum swinging
def animated_pendulum():
    def pendulum_position(theta: float):
        return (l*np.sin(theta), -l*np.cos(theta))

    # create figure
    fig = plt.figure()
    axis = fig.add_subplot(aspect='equal')
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1.25, 0.25)
    axis.grid()

    x0, y0 = pendulum_position(theta0)
    line, = axis.plot([0, x0], [0, y0], linewidth=2, color='k')
    circle = patches.Circle(pendulum_position(theta0), radius=0.05, fc='r', zorder=3)
    blob = axis.add_patch(circle)

    def animate(frame: int):
        x, y = pendulum_position(theta[frame])
        line.set_data([0, x], [0, y])
        circle.center = (x, y)
        return line, blob,

    anim = animation.FuncAnimation(fig, animate, frames=len(t))
    ffmpeg_writer = animation.FFMpegWriter(fps=30)
    anim.save('pendulum.mp4', writer=ffmpeg_writer)

# Animate Everything
def animated_plots():
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

    # theta, theta_dot vs time
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_xlim(0, 5)
    ax0.set_ylim(-100, 100)
    ax0.set_ylabel(r'$\theta (^\circ)$, $\dot \theta (^\circ/s)$')
    ax0.legend([r'$\theta$', r'$\dot \theta$'])
    ax0.grid()

    theta_curve, = ax0.plot(t[0], theta_deg[0], 'b')
    theta_dot_curve, = ax0.plot(t[0], theta_dot_deg[0], 'r')

    # phase diagram
    ax1 = fig.add_subplot(gs[1,0])
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.set_xlabel(r'$\theta (^\circ)$')
    ax1.set_ylabel(r'$\dot \theta (^\circ/s)$')
    ax1.grid()

    phase_curve, = ax1.plot(theta_deg[0], theta_dot_deg[0], 'b')
    phase_dot, = ax1.plot(theta_deg[0], theta_dot_deg[0], 'ro')

    # pendulum swinging
    def pendulum_position(theta: float):
        return (l*np.sin(theta), -l*np.cos(theta))

    ax2 = fig.add_subplot(gs[:,1])
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1.5, 0.5)

    x0, y0 = pendulum_position(theta0)
    line, = ax2.plot([0, x0], [0, y0], linewidth=2, color='k')
    circle = patches.Circle(pendulum_position(theta0), radius=0.05, fc='r', zorder=3)
    blob = ax2.add_patch(circle)

    def animate(frame: int):
        theta_curve.set_data(t[:frame+1], theta_deg[:frame+1])
        theta_dot_curve.set_data(t[:frame+1], theta_dot_deg[:frame+1])

        phase_curve.set_data(theta_deg[:frame+1], theta_dot_deg[:frame+1])
        phase_dot.set_data(theta_deg[frame:frame+1], theta_dot_deg[frame:frame+1])

        x, y = pendulum_position(theta[frame])
        line.set_data([0, x], [0, y])
        circle.center = (x, y)

        return theta_curve, theta_dot_curve, phase_curve, phase_dot, line, blob,

    anim = animation.FuncAnimation(fig, animate, frames=len(t), blit=True)
    ffmpeg_writer = animation.FFMpegWriter(fps=30)
    anim.save('animation.mp4', writer=ffmpeg_writer)


# static_pqvt_plot()
# animated_pqvt_plot()
# static_phase_space_plot()
# animated_phase_space_plot()
# animated_pendulum()
# animated_plots()

