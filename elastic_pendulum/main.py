from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from numpy import ndarray, array, cos, sin, deg2rad, linspace
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


class Bob:
    def __init__(self, mass: float):
        self.mass = mass

class Rod:
    def __init__(self, spring_constant: float, length: float):
        self.spring_constant = spring_constant
        self.length = length


class Pendulum:
    def __init__(self, bob: Bob, rod: Rod, damping_coefficient: float, gravity_acceleration: float=9.81) -> None:
        self.bob: Bob = bob
        self.rod: Rod = rod
        self.gravity_acceleration: float = gravity_acceleration
        self.damping_coefficient: float = damping_coefficient
        self.resonant_frequency: float = self.gravity_acceleration / self.rod.length

    def compute_statevector(self, _, state: ndarray) -> ndarray:
        """Calculate the derivatives of the angular position and velocity"""
        theta: float = state[0] # angular position
        omega: float = state[1] # angular velocity
        r_len: float = state[2] # length of spring/rod
        r_vel: float = state[3] # rate of change of spring/rod length

        # angular acceleration
        theta_acc: float = (-1/r_len) * ( (self.gravity_acceleration * sin(theta)) + (2 * r_vel * omega))
        # acceleration of rod/spring
        r_acc: float = (r_len * omega**2) + (self.gravity_acceleration * cos(theta)) - ( (self.rod.spring_constant / self.bob.mass) * (r_len - self.rod.length) )

        return array([omega, theta_acc, r_vel, r_acc])

    def position(self, theta: float, length: float) -> tuple[float, float]:
        """Calculate the position of the bob"""
        return ( length * sin(theta), - length * cos(theta) )


class Solver:
    """Solves the pendulum's equations of motion using numerical integration"""
    def __init__(self, pendulum: Pendulum, initial_conditions: ndarray, time_span: list, time_eval: ndarray) -> None:
        self.pendulum: Pendulum = pendulum
        self.initial_conditions: ndarray = initial_conditions
        self.time_span: list = time_span
        self.time_eval: ndarray = time_eval
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
    def __init__(self, solver: Solver) -> None:
        self.solver: Solver = solver

        self.t: ndarray = self.solver.solution.t
        self.theta: ndarray = self.solver.solution.y[0]
        self.omega: ndarray = self.solver.solution.y[1]
        self.r_len: ndarray = self.solver.solution.y[2]
        self.r_vel: ndarray = self.solver.solution.y[3]

        # Labels for the plots
        self.theta_label: str = r'$\theta (^\circ)$'
        self.omega_label: str = r'$\omega (^\circ/s)$'

    def animated_plots(self, filename: str, save_as_gif: bool = True) -> None:
        """Plots the pendulum's dynamics"""
        fig = plt.figure(dpi=(1920/16))
        fig.set_size_inches(19.20, 10.80)
        fig.suptitle('Elastic Pendulum')
        gs = GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        # position vs. time
        axes0: Axes = fig.add_subplot(gs[1,:])
        axes0.set_ylabel(self.theta_label + " " + self.omega_label)
        axes0.set_xlabel('Time (s)')
        theta_curve: Line2D = axes0.plot(self.t, self.theta, 'b')[0]
        omega_curve: Line2D = axes0.plot(self.t, self.omega, 'r')[0]
        axes0.set_title(r'$Angular Position (\theta) & Velocity (\omega) vs. Time$')
        axes0.legend([self.theta_label, self.omega_label])
        axes0.grid()

        # phase space
        axes1: Axes = fig.add_subplot(gs[0, 0])
        axes1.set_ylabel(self.omega_label)
        axes1.set_xlabel(self.theta_label)
        axes1.set_title(r'$Angular velcity (\omega) vs. Angular Displacement (\theta): Phase Space$')
        axes1.grid()
        phase_curve: Line2D = axes1.plot(self.theta, self.omega, 'b')[0]
        phase_dot: Line2D = axes1.plot(self.theta, self.omega, 'ro')[0]

        # pendulum animation
        axes2: Axes = fig.add_subplot(gs[0, 2])
        axes2.set_xlim(-1, 1)
        axes2.set_ylim(-1.5, 0.5)
        axes2.set_title('Elastic Pendulum Animation')

        theta_init: float = self.solver.initial_conditions[0]
        x0, y0 = self.solver.pendulum.position(theta=theta_init, length=self.r_len[0])

        line: Line2D = axes2.plot([0, x0], [0, y0], linewidth=2, color='k')[0]
        circle: Circle = Circle(self.solver.pendulum.position(theta_init, length=y0), radius=0.05, color='r', zorder=3)
        bob: Patch = axes2.add_patch(circle)

        def animate(frame: int) -> tuple:
            theta_curve.set_data(self.t[:frame+1], self.theta[:frame+1])
            omega_curve.set_data(self.t[:frame+1], self.omega[:frame+1])

            phase_curve.set_data(self.theta[:frame+1], self.omega[:frame+1])
            phase_dot.set_data(self.theta[frame:frame+1], self.omega[frame:frame+1])

            x, y = self.solver.pendulum.position(theta=self.theta[frame], length=self.r_len[frame])
            circle.center = (x, y)
            line.set_data([0, x], [0, y])

            return theta_curve, omega_curve, phase_curve, phase_dot, line, bob

        # Animate the plots
        if save_as_gif:
            animation: FuncAnimation = FuncAnimation(fig, animate, repeat=True, frames=len(self.t), blit=True)
            pillow_writer: PillowWriter = PillowWriter(fps=30)
            animation.save(filename, writer=pillow_writer)
        else:
            animation: FuncAnimation = FuncAnimation(fig, animate, repeat=True, frames=len(self.t), blit=True)
            ffmpeg_writer: FFMpegWriter = FFMpegWriter(fps=30)
            animation.save(filename, writer=ffmpeg_writer)


if __name__ == '__main__':
    # Initial conditions: theta, omega, r, v
    theta_init: float = deg2rad(30) # initial angular displacement
    omega_init: float = 0.0 # initial angular velocity
    r_len_init: float = 1.0 * 1.20 # initial stretch of spring/rod
    r_vel_init: float = 0.0 # initial tangential velocity of spring/rod
    initial_conditions: ndarray = array([theta_init, omega_init, r_len_init, r_vel_init])

    # Time range: 0 to 10 seconds with 1000 points
    time_span: list = [0, 10]
    time_eval: ndarray = linspace(*time_span, 230)

    # create elastic pendulum
    elastic_pendulum: Pendulum = Pendulum(
            bob=Bob(mass=1.0),
            rod=Rod(spring_constant=34.0, length=0.5), # length l at equilibrium
            damping_coefficient=0.0,
            gravity_acceleration=9.81
        )
    solver: Solver = Solver(elastic_pendulum, initial_conditions, time_span, time_eval)
    solver.solve()

    plotter: PlotDynamics = PlotDynamics(solver)
    plotter.animated_plots('elastic_pendulum.gif', True)

