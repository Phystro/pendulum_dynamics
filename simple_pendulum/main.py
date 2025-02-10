from matplotlib.patches import Circle, Patch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


class Bob:
    def __init__(self, mass: float):
        self.mass = mass


class Rod:
    def __init__(self, length: float):
        self.length = length


class Pendulum:
    def __init__(self, bob: Bob, rod: Rod, damping_coeff: float, gravity_acc: float=9.81):
        self.bob = bob
        self.rod = rod
        self.damping_coeff = damping_coeff
        self.gravity_acc = gravity_acc


    def compute_statevector(self, _, state: np.ndarray) -> np.ndarray:
        """Calculate the derivatives of angular position and velocity"""
        theta, omega = state
        # damping_term = (self.damper.damping_coefficient /
        #                (self.bob.mass * self.rod.length**2)) * omega
        damping_factor: float = (self.damping_coeff/self.bob.mass) * omega
        domega_dt: float = -(self.gravity_acc/self.rod.length) * np.sin(theta) - damping_factor
        return np.array([
            omega,
            domega_dt
            ])

    def position(self, theta: float) -> tuple[float, float]:
        """Calculate the position of the bob"""
        return (self.rod.length*np.sin(theta), -self.rod.length*np.cos(theta))


class Solver:
    """Solves the pendulum's equations of motion using numerical integration"""
    def __init__(self, pendulum: Pendulum, initial_conditions: np.ndarray, time_span: list, time_eval: np.ndarray) -> None:
        self.pendulum = pendulum
        self.initial_conditions = initial_conditions
        self.time_span = time_span
        self.time_eval = time_eval
        self.solution: OdeResult = None

    def solve(self) -> None:
        """Integrates the system using scipy.integrate.solve_ivp"""
        self.solution: OdeResult = solve_ivp(
                fun=self.pendulum.compute_statevector,
                t_span=self.time_span,
                y0=self.initial_conditions,
                t_eval=self.time_eval
            )
        print(self.solution)


class PlotDynamics:
    """Generate plots from the solver's results"""
    def __init__(self, solver: Solver) -> None:
        self.solver = solver

        if self.solver.solution is None:
            # raise Exception('Solution not found')
            raise RuntimeError("Run the solver first before plotting")

        self.t: np.ndarray = self.solver.solution.t
        self.theta: np.ndarray = np.rad2deg(self.solver.solution.y[0])
        self.omega: np.ndarray = np.rad2deg(self.solver.solution.y[1])

        # Labels for the plots
        self.theta_label: str = r'$\theta (^\circ)$'
        self.omega_label: str = r'$\omega (^\circ/s)$'


    def plot_position_v_time(self) -> None:
        """Plot the angular position of the bob vs. time"""
        fig, axes = plt.subplots(figsize=(10, 5))
        plt.plot(self.t, self.theta, 'r', linewidth=2, label=self.theta_label)
        plt.plot(self.t, self.omega, 'b', linewidth=2, label=self.omega_label)
        plt.title('Simple Pendulum Dynamics')
        plt.xlabel('Time (s)')
        plt.ylabel(self.theta_label + ', ' + self.omega_label)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_phase_diagram(self) -> None:
        """plot of phase diagram of theta, theta_dot"""
        fig, axes = plt.subplots(figsize=(10, 5))
        plt.plot(self.theta, self.omega, 'b')
        plt.title('Simple Pendulum: Phase Diagram')
        plt.xlabel(self.theta_label)
        plt.ylabel(self.omega_label)
        plt.grid()
        plt.show()

    def plot_phase_space_position_v_time(self) -> None:
        """Combined plots of phase space diagram and angular position vs. time"""
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.suptitle('Simple Pendulum Dynamics')
        axes[0].plot(self.theta, self.omega)
        axes[1].plot(self.t, self.theta)
        axes[1].plot(self.t, self.omega)
        plt.show()

    def animated_plots(self, filename: str, save_as_gif: bool = True) -> None:
        """Animated plots of phase diagram, angular position vs. time and pendulum animation"""
        fig = plt.figure(dpi=(1920/16))
        fig.set_size_inches(19.20, 10.80)
        fig.suptitle('Simple Pendulum Dynamics')
        gs = GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        # position vs. time
        # axes0: Axes = fig.add_subplot(gs[0, 0])
        axes0: Axes = fig.add_subplot(gs[1,:])
        axes0.set_ylabel(self.theta_label)
        axes0.set_xlabel('Time (s)')
        theta_curve: Line2D = axes0.plot(self.t, self.theta, 'b')[0]
        omega_curve: Line2D = axes0.plot(self.t, self.omega, 'r')[0]
        axes0.set_title('Angular Position & Velocity vs. Time')
        axes0.legend([self.theta_label, self.omega_label])
        axes0.grid()

        # phase space
        axes1: Axes = fig.add_subplot(gs[0, 0])
        axes1.set_ylabel(self.omega_label)
        axes1.set_xlabel(self.theta_label)
        axes1.set_title('Phase Space Diagram')
        axes1.grid()
        phase_curve: Line2D = axes1.plot(self.theta, self.omega, 'b')[0]
        phase_dot: Line2D = axes1.plot(self.theta, self.omega, 'ro')[0]

        # pendulum animation
        # axes2: Axes = fig.add_subplot(gs[:,1])
        axes2: Axes = fig.add_subplot(gs[0, 2])
        axes2.set_xlim(-1, 1)
        axes2.set_ylim(-1.5, 0.5)
        axes2.set_title('Simple Pendulum Animation')

        theta_init: float = self.solver.initial_conditions[0]
        x0, y0 = self.solver.pendulum.position(theta=theta_init)

        line: Line2D = axes2.plot([0, x0], [0, y0], linewidth=2, color='k')[0]
        circle: Circle = Circle(self.solver.pendulum.position(theta_init), radius=0.05, color='r', zorder=3)
        bob: Patch = axes2.add_patch(circle)

        def animate(frame: int) -> tuple:
            theta_curve.set_data(self.t[:frame+1], self.theta[:frame+1])
            omega_curve.set_data(self.t[:frame+1], self.omega[:frame+1])

            phase_curve.set_data(self.theta[:frame+1], self.omega[:frame+1])
            phase_dot.set_data(self.theta[frame:frame+1], self.omega[frame:frame+1])

            x, y = self.solver.pendulum.position(theta=np.deg2rad(self.theta[frame]))
            circle.center = (x, y)
            line.set_data([0, x], [0, y])

            return theta_curve, omega_curve, phase_curve, phase_dot, line, bob

        if save_as_gif:
            animation: FuncAnimation = FuncAnimation(fig, animate, repeat=True, frames=len(self.t), blit=True)
            pillowwriter: PillowWriter = PillowWriter(fps=30, metadata=dict(artist='PhyTensor'), bitrate=1800)
            animation.save(filename, writer=pillowwriter)
        else:
            animation: FuncAnimation = FuncAnimation(fig, animate, frames=len(self.t), blit=True)
            ffmpegwriter: FFMpegWriter = FFMpegWriter(fps=30)
            animation.save(filename, writer=ffmpegwriter)


if __name__ == '__main__':
    # initial conditions: initial angular displacement, initial angular velocity
    theta_init: float = np.deg2rad(30)
    omega_init: float = 0.0 # initial angular velocity
    initial_conditions: np.ndarray = np.array([theta_init, omega_init])

    # Time range: 0 to 5 seconds with 1000 points
    time_span: list = [0, 10]
    time_eval: np.ndarray = np.linspace(*time_span, 130)


    # Simple undamped pendulum
    simple_undamped_pendulum: Pendulum = Pendulum(
        rod=Rod(length=1.0),
        bob=Bob(mass=1.0),
        damping_coeff=0.0
    )

    # Create and run solver
    solver: Solver = Solver(simple_undamped_pendulum, initial_conditions, time_span, time_eval)
    solver.solve()

    # Generate and display plots
    plotter: PlotDynamics = PlotDynamics(solver)
    plotter.animated_plots(save_as_gif=True, filename='simple_undamped_pendulum.gif')

    # Simple damped pendulum
    simple_damped_pendulum: Pendulum = Pendulum(
        rod=Rod(length=1.0),
        bob=Bob(mass=1.0),
        damping_coeff=0.5
    )

    # Create and run solver
    solver: Solver = Solver(simple_damped_pendulum, initial_conditions, time_span, time_eval)
    solver.solve()

    # Generate and display plots
    plotter: PlotDynamics = PlotDynamics(solver)
    plotter.animated_plots(save_as_gif=True, filename='simple_damped_pendulum.gif')

