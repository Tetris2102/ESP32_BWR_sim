from lcd_i2c import LCD
from machine import SoftI2C, Pin
from time import sleep

def mean(list):
    """Computes mean of a list"""
    return sum(list) / len(list)

def sign(num):
    """Determines sign of a number"""
    if num > 0:
        return 1
    elif num < 0:
        return -1
    return 0

def clip(num, min_val, max_val):
    """Clips a number to maximum or minimum value if not within range"""
    if num < min_val:
        return min_val
    if num > max_val:
        return max_val
    return num

class ReactorCore:
    
    def __init__(self):
        self.rods_pos = 0.0  # Rods position in percents, 1 is full out (full power), 0 is full in (zero power)
        self.initial_power = 0.0  # kW
        self.y0 = [self.initial_power] + [1.0] * 6  # Power + 6 precursor groups
        self.beta = 0.0065  # For U-235
        self.lambda_i = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
        self.beta_i = [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]
        self.neutron_lifetime = 50e-6

        self.power = self.initial_power
        self.power_history = [self.initial_power] * 4  # Added this line

        self.rho_min = -0.003
        self.rho_max = 0.007

        # rods_speed must be a multiple of rods_step
        self.rods_step = 0.001  # Minimum rod step
        self.rods_speed = 5 * self.rods_step  # Fraction of 1 per second
        self.rods_update_count = 0

        self.autopilot_power = 500.0  # kW

        self.time_step = 1.0  # Time step of simulation, 1.0 corresponds to real time, 10.0 to 10x speed

        self.is_scram = False

    def reactivity(self, rods_pos=None):
        if rods_pos is None:
            rods_pos = self.rods_pos
        return (self.rho_max - self.rho_min) * rods_pos + self.rho_min

    def period(self):
        if len(self.power_history) > 3:
            dp_dt = (self.power_history[3] - self.power_history[2]) / self.time_step
        else:
            return "inf"

        if self.power > 0.0 and dp_dt > 1e-4:
            e = 2.7183
            period = e * self.power / dp_dt
            if period < 2000:
                return period

        return "inf"

    def point_kinetics_full(self, t, y):
        """Full point kinetics equations for a U-235 reactor with multiple precursor groups."""
        # Add at start of point_kinetics_full:
        power = float(y[0])  # Force explicit float conversion
        precursors = [float(p) for p in y[1:]]  # Ensure proper precision

        # Temperature feedback
        temp_feedback = -6.5e-6 * power
        
        rho = self.reactivity() + temp_feedback
        beta_total = sum(self.beta_i)
        
        # Point kinetics equations
        dp_dt = ((rho - beta_total) / self.neutron_lifetime) * power
        for i in range(len(self.lambda_i)):
            dp_dt += self.lambda_i[i] * precursors[i]
        
        # Precursor equations (one per group)
        dc_dt = []
        for i in range(len(self.lambda_i)):
            dc_dt.append((self.beta_i[i] / self.neutron_lifetime) * power - self.lambda_i[i] * precursors[i])
        
        return [dp_dt] + dc_dt

    def rk4_step(self, t, y, dt):
        """Perform a single RK4 step with better numerical stability"""
        k1 = self.point_kinetics_full(t, y)
        k2 = self.point_kinetics_full(t + dt/2, [y[i] + dt/2 * k1[i] for i in range(len(y))])
        k3 = self.point_kinetics_full(t + dt/2, [y[i] + dt/2 * k2[i] for i in range(len(y))])
        k4 = self.point_kinetics_full(t + dt, [y[i] + dt * k3[i] for i in range(len(y))])
        
        # Calculate new values with protection against extreme values
        new_y = []
        for i in range(len(y)):
            delta = dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            if abs(delta) > 1e10:  # Prevent extreme jumps
                delta = sign(delta) * 1e10
            new_val = y[i] + delta
            new_y.append(new_val)
        
        return new_y

    def solve_power(self, time):
        """Solve with substeps while maintaining the same overall time_step"""

        substeps = max(1, int(self.time_step / 0.007))  # Aim for ~50ms substeps
        h = self.time_step / substeps  # Actual RK4 step size
        
        for _ in range(substeps):
            self.y0 = self.rk4_step(time, self.y0, h)
            time += h
            
            # Physical constraints
            self.y0[0] = max(0, self.y0[0])  # Power can't be negative
            for i in range(1, len(self.y0)):
                self.y0[i] = max(0, self.y0[i])  # Precursors can't be negative
        
        self.power_history.append(self.y0[0])
        if len(self.power_history) > 4:
            self.power_history.pop(0)
        
        self.power = mean(self.power_history)
            
        return self.power

    def increment_rods(self, goal_rods_pos):
        """Moves rods toward goal_rods_pos with minimum step and maximum speed constraints"""
        # Calculate maximum allowed movement this step
        max_possible_move = self.rods_speed * self.time_step
        
        # Calculate desired movement direction and distance
        movement_needed = goal_rods_pos - self.rods_pos
        movement_sign = sign(movement_needed)
        movement_distance = abs(movement_needed)
        
        # Determine actual step size
        if movement_distance < self.rods_step:
            # Too small to move unless at boundary
            if goal_rods_pos in (0.0, 1.0):
                self.rods_pos = goal_rods_pos
        else:
            # Move by either the full needed distance or rods_speed * time_step
            step_size = movement_sign * max(
                min(movement_distance, max_possible_move),
                self.rods_step
            )
            step_size = step_size // self.rods_step * self.rods_step
            self.rods_pos += step_size
        
        # Ensure we stay within bounds
        self.rods_pos = clip(self.rods_pos, 0.0, 1.0)
    
    def move_rods(self, goal_rods_pos):
        """Moves rods gradually and prevents movement in case of SCRAM"""

        if goal_rods_pos >= 0 and goal_rods_pos <= 1:
            self.increment_rods(goal_rods_pos)
        elif goal_rods_pos > 1:
            self.increment_rods(1)
        elif goal_rods_pos < 0:
            self.increment_rods(0)

    def autopilot(self, goal_pwr=None):

        if self.is_scram == False:

            if goal_pwr == None:
                goal_pwr = self.autopilot_power

            autopilot_window = 0.005 * goal_pwr  # 0.5% tolerance window (tighter control)
            real_pwr = mean(self.power_history)

            # If within tolerance, don't adjust
            if abs(real_pwr - goal_pwr) < autopilot_window:
                return

            # Calculate normalized error (0 to 1)
            diff = abs(real_pwr - goal_pwr)
            normalized_diff = min(diff / goal_pwr, 1.0)

            # Scale step between rods_step (min) and rods_speed * time_step (max)
            max_step = self.rods_speed * self.time_step
            step = self.rods_step + (max_step - self.rods_step) * normalized_diff
            step = round(step, 4)

            # Adjust rods (direction corrected)
            if real_pwr > goal_pwr:
                # Power too high → insert rods (INCREASE rods_pos)
                new_pos = min(1.0, self.rods_pos - step)
                self.move_rods(new_pos)
            else:
                # Power too low → withdraw rods (DECREASE rods_pos)
                new_pos = max(0.0, self.rods_pos + step)
                self.move_rods(new_pos)

    def check_SCRAM(self, cond_on, cond_off=False):
        """
        Checks for and handles SCRAM using given conditions
        for initiating and finishing SCRAM
        """

        if cond_on or self.is_scram:
            self.is_scram = True
            self.move_rods(0.0)

            if cond_off:
                self.is_scram = False

def print_vals_2004(lcd: object, vals: list):
    """
    Prints strings from a given list in two columns on a 2004 I2C LCD
    """
    
    if len(vals) > 8:
        raise Exception("List too long to print out")
    
    lcd.set_cursor(col=0, row=0)
    
    for i in range(len(vals)):
        if i < 4:
            lcd.set_cursor(col=0, row=i)
            lcd.print(vals[i])
        else:
            lcd.set_cursor(col=10, row=i%4)
            lcd.print(vals[i])
    
    lcd.set_cursor(row=0, col=0)

def format_float(value, decimals=3):
    """
    Formats float to have the specified number of decimal places
    """
    
    if value is None:
        return None
    
    if value > 1.0:
        value = round(value, 0)
    else:
        value = round(value, decimals)
    

# 2004 LCD
scl_2004 = Pin(13)
sda_2004 = Pin(14)
I2C_ADDR = 0x27     # DEC 39, HEX 0x27
NUM_ROWS = 4
NUM_COLS = 20

i2c = SoftI2C(scl=scl_2004, sda=sda_2004, freq=800000)
lcd = LCD(addr=I2C_ADDR, cols=NUM_COLS, rows=NUM_ROWS, i2c=i2c)

lcd.begin()

reactor = ReactorCore()

lcd.clear()

time_now = 0
reactor.time_step = 1.0
reactor.initial_power = 10

while True:

    power = reactor.solve_power(time=time_now)

    period = reactor.period()

    reactor.autopilot(500.0)

    space = 10 * ' '
#     vals = ["Pow:" + power_str + '   ', "Rods:" + rods_str + '  ', "Per:" + period_str + '   ', space, space, space, space, space]
    vals = [f"Pow:{power:.1f}   ", f"Rods:{reactor.rods_pos:.1f}  ", f"Per:{period}   ", space, space, space, space, space]
    
    print_vals_2004(lcd, vals)
    
    time_now += 1

    sleep(0.9)
    
#     space = 10 * ' '
#     spaces = ['', '', space]
#     print_vals_2004(lcd, spaces)

