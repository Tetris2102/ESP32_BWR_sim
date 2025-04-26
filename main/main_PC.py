from math import sqrt

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
        self.rods_pos = 1.0  # Rods position in percents, 1 is full out (full power), 0 is full in (zero power)
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
        self.rods_step = 0.005  # Minimum rod step
        self.rods_speed = 4 * self.rods_step  # Fraction of 1 per second
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

            print(step)

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


class ReactorSystems:

    def __init__(self):

        # Simulation constants
        self.time_step = 1.0  # Time step of simulation
        self.eq_k = 0.7  # How fast pressure and temperature equalise, e.g. when vlvA_main is opened

        # Pressure (average), kPa:
        self.press_ambient = 101.325
        self.pressA_1_default = self.press_ambient  # Standard atmospheric pressure
        self.pressA_1 = self.pressA_1_default  # Pressure at core (first loop)
        self.pressA_2 = self.press_ambient  # Pressure at steam generator (first loop)
        self.pressA_3 = self.press_ambient  # Pressure at pressuriser (first loop)
        self.pressB_1 = self.press_ambient  # Pressure before turbine (second loop)
        self.pressB_2 = self.press_ambient  # Pressure after turbine (second loop)
        # self.pressB_3 = self.press_ambient  # Pressure after condenser (second loop)  # No B_3, B_2 used instead

        # Temperature (average), degrees C:
        self.temp_ambient = 20.0  # Ambient temperature, C
        self.tempA_1 = self.temp_ambient  # Temperature before valve (first loop)
        self.tempA_2 = self.temp_ambient  # Temperature at steam generator (first loop)
        self.tempA_3 = self.temp_ambient  # Temperature at pressuriser (first loop)
        self.tempB_1 = self.temp_ambient  # Temperature before turbine (second loop)
        self.tempB_2 = self.temp_ambient  # Temperature after turnine (second loop)
        # self.tempB_3 = self.temp_ambient  # Temperature after condenser (second loop)  # No B_3, B_2 used instead
        self.temp_core = self.temp_ambient  # Temperature of core, C

        # Water flow, m^3/s:
        self.flowA = 0.0  # Flow through main valve of first loop
        self.flowB = 0.0  # Flow through main valve of second loop
        self.flow_from_A_4 = 0.0  # Flow from A_4 to A_2
        self.flow_to_A_4 = 0.0  # Flow to A_4 from lake

        # Volume, m^3:
        # Coolant is approximated as steam only
        self.volA_1_default = 40.0  # Default value used later to adjust pressure for volA_1
        self.volA_1 = self.volA_1_default  # Volume of coolant at core part of first loop
        self.volA_2_default = 40.0
        self.volA_2 = self.volA_2_default  # Volume of coolant in stean generator part of first loop
        self.volA_4 = 200.0  # Volume of coolant in reserve tank (first loop)
        self.volA_3_default = 50.0
        self.volA_3 = self.volA_3_default  # Volume of coolant in pressuriser (first loop)
        self.volB_1_default = 70.0
        self.volB_1 = self.volB_1_default  # Volume of coolant after pump and before vlvB_main (second loop)
        self.volB_2_default = 100.0
        self.volB_2 = self.volB_2_default  # Volume of coolant after vlvB_main and before pump (second loop)

        self.vol_relief = 0.0  # Volume of coolant in relief tank

        # Valves:
        self.vlvA_main = True  # Valve before steam generator
        self.vlvA_relief = False  # Emergency relief valve
        self.vlvA_4 = False  # Reserve tank valve
        self.vlvA_3 = False  # Pressuriser valve

        self.vlvB_main = True  # Valve before turbine
        self.vlvB_relief = False
        self.vlvB_2 = False  # Valve after turbine
        self.vlvB_4 = False  # Valve before pump for filling reserve tank for first loop

        self.vlv_empty_relief = False  # Valve to empty relief tank

        # Pumps:
        self.pumpA = False
        self.pumpA_speed = 12  # m^3/s
        self.pumpB = False
        self.pumpB_speed = 15  # m^3/s
        self.pumpA_4 = False  # Pump to fill reserve water tank
        self.pumpA_4_speed = 5  # m^3/s

        # Volumes of tanks, m^3:
        self.VOLUME_A_1 = 40.0  # Volume of first part of first loop
        self.VOLUME_A_2 = 40.0 # Volume of second part of second loop
        self.VOLUME_A = self.VOLUME_A_1 + self.VOLUME_A_2  # Volume of first loop
        self.VOLUME_A_3 = 30.0  # Volume of pressuriser
        self.VOLUME_A_4 = 40.0 # Volume of reserve coolant tank
        self.VOLUME_B_1 = 100.0  # Volume of first part of second loop
        self.VOLUME_B_2 = 70.0  # Volume of second part of second loop
        self.VOLUME_B = 170.0  # Volume of second loop
        self.VOLUME_RELIEF = 200.0  # Volume of relief tank

        # Physical constants:
        self.WORKING_PRESSURE_A = 7600.0  # Typical for BWR normal operating pressure (first loop), kPa (75 atmospheres)
        self.WORKING_PRESSURE_B = 6200.0  # GUESS, Normal operating pressure (second loop), kPa
        self.WATER_DENSITY = 1000.0  # kg/m^3
        self.HEAT_CAPACITY_WATER = 4.184  # J/(kg*K)
        # self.HEAT_TRANSFER_COEFF = 5000  # W/(m²*K) - typical for water-cooled reactors
        self.MASS_CORE = 100000  # kg
        self.HEAT_CAPACITY_CORE = 247  # J/(kg*K) - uranium dioxide specific heat

        # Flow constants, m^3/s:
        self.MAX_RELIEF_FLOW = 12.0  # Maximum flow of coolant into relief tank for either loop
        self.MAX_FLOW_A = 20.0
        self.MAX_FLOW_A_4 = 7.0
        self.MAX_FLOW_B = 15.0
        # self.RELIEF_FLOW_RATE = 25.0

        self.MAX_PRESSURE = 10000.0  # Max pressure for both loops, kPa
        self.TURBINE_EFFICIENCY = 0.85  # Efficiency of water turbine times efficiency of electric generator
        self.launch_speed_turbine = 50.0 / self.time_step  # One second to increase turbine power by 50 kW
        # self.SURFACE_AREA_CORE = 200  # m^2 - approximate heat transfer surface area
        # self.SURFACE_AREA_A = 200   # m^2 - used for calculating heat loss
        # self.PUMP_MAX_FLOW = 12         # Flow created by pumps at maximum power, m^3/s

        # Memory values:
        self.delay_pressA_3 = 0  # Used to create smooth equalisation of pressures between
                                 # the first loop and pressuriser
        self.delay_pumpA = 4.0 / self.time_step  # How many function calls are required to fully launch pump in 4 seconds
        self.delay_pumpB = 4.0 / self.time_step
        self.delay_pumpA_4 = 2.0 / self.time_step
        self.marker_pumpA = self.pumpA  # Used to track shutdown of pumps
        self.marker_pumpB = self.pumpB
        self.marker_pumpA_4 = self.pumpA_4
        self.delay_count_pumpA = 0  # Are used to manage pump launching
        self.delay_count_pumpB = 0
        self.delay_count_pumpA_4 = 0

        # Faults:
        self.A_OP = False  # First loop overpressure, pressure loss
        self.A_3_OP = False  # Pressuriser overpressure, pressure loss
        self.MELTDOWN = False  # Core meltdown
        self.B_OP = False  # Second loop overpressure, pressure loss

        # Other:
        self.turbine_power = 0  # Turbine generator output power, kW
        self.catastrophe = False  # Is used to raise catastrophe situation
        self.catastrophe_log = ""  # Is used to give catastrophe information

    def pressure(self, temp, vol_water, vol_total):
        """
        Calculates water pressure using a more accurate model for water-vapor mixture
        
        temp - temperature of mixture, degrees C
        vol_water - volume of all water if it were liquid, m^3
        vol_total - volume of the system, m^3
        
        Returns pressure in kPa
        """
        # Check if temperature is above boiling point
        if temp < 100:
            # Below boiling point - just hydrostatic pressure
            return self.press_ambient  # Standard atmospheric pressure
        
        # Calculate saturation pressure at given temperature using Antoine equation
        # Constants for water (100-374°C)
        A = 8.14019
        B = 1810.94
        C = 244.485
        
        # Saturation pressure in mmHg
        P_sat_mmHg = 10**(A - (B / (temp + C)))
        # Convert to kPa (1 mmHg = 0.133322 kPa)
        P_sat = P_sat_mmHg * 0.133322
        
        # Calculate vapor volume
        liquid_volume = min(vol_water, vol_total)  # Can't have more liquid than total volume
        vapor_volume = vol_total - liquid_volume
        
        if vapor_volume <= 0:
            # All liquid, pressure is hydrostatic plus some margin
            return max(self.press_ambient, P_sat)
        
        # Calculate vapor mass using density at saturation
        # Approximate vapor density using ideal gas law at saturation pressure
        vapor_density = 0.018 * P_sat / (8.314 * (temp + 273.15))  # kg/m³
        vapor_mass = vapor_density * vapor_volume
        
        # Calculate total pressure
        # If system is at saturation, return saturation pressure
        # If system has excess vapor, calculate pressure using ideal gas law
        if liquid_volume > 0:
            # Two-phase mixture at saturation
            return P_sat
        else:
            # Superheated steam only
            # Use ideal gas law with steam properties
            mol = vapor_mass / 0.018  # 18 g/mol for water
            pressure = mol * 8.314 * (temp + 273.15) / (vol_total * 1000)
            return max(pressure, P_sat)

    def compute_equalised_press(self, p1, v1, p2, v2):
        """
        Computes pressure of two equalised tanks

        p1, p2 - pressures in tanks before connecting, kPa
        v1, v2 - volumes of tanks, m^3
        """

        press = (p1 * v1 + p2 * v2) / (v1 + v2)

        return press

    # SUGGESTIONS:
    # Replace self.pumpA_speed -> self.pumpA_nom_speed  # Nominal speed of pumpA
    # Replace self.pumpB_speed -> self.pumpB_nom_speed  # Nominal speed of pumpB
    # If meltdown occurs, lock the position of rods by not allowing them to change
    # Check if self.pressA_1_default is needed anymore
    # If pump is off, flow should be 0
    # Review catastrophe_log messages to make sure they fit display

    def simulate_systems(self, power):
        """
        Simulates all reactor systems and
        updates corresponding variables

        power - reactor thermal power, kW
        self.time_step - time step of simulation, seconds
        """

        self.check_catastrophe()
        self.update_core(power)
        self.update_A()
        self.update_B()
        self.update_turbine()
        self.update_relief()
        self.update_reserve()
        self.catch_out_of_range()

    def update_A(self):
        """
        Updates temperature, pressure and volume of coolant in A_1, A_2 and pressuriser (A_3)
        """

        # Update pressure at A_1 and A_2
        if self.vlvA_main and self.MELTDOWN == False:
            pressA_1 = self.pressure(self.tempA_1, self.volA_1, self.VOLUME_A)
            pressA_2 = self.pressure(self.tempA_2, self.volA_2, self.VOLUME_A)
        else:
            pressA_1 = self.pressure(self.tempA_1, self.volA_1, self.VOLUME_A_1)
            pressA_2 = self.pressure(self.tempA_2, self.volA_2, self.VOLUME_A_2)

        # Update pressure at pressuriser
        if self.vlvA_3:
            pressA_3 = self.pressure(self.tempA_3, self.volA_3, self.VOLUME_A_3 + self.VOLUME_A_1)
        else:
            pressA_3 = self.pressure(self.tempA_3, self.volA_3, self.VOLUME_A_3)

        self.pressA_1 += self.eq_k * (pressA_1 - self.pressA_1)
        self.pressA_2 += self.eq_k * (pressA_2 - self.pressA_2)
        self.pressA_3 += self.eq_k * (pressA_3 - self.pressA_3)

        # Equalise pressures between A_3 and A_1
        if self.vlvA_3:

            press3 = self.compute_equalised_press(self.pressA_3, self.volA_3, self.pressA_1, self.VOLUME_A_1)

            self.pressA_1 += self.eq_k * (press3 - self.pressA_1)
            self.pressA_3 += self.eq_k * (press3 - self.pressA_3)

        if self.vlvA_main and self.MELTDOWN == False:  # Meltdown does not allow flow of coolant

            # Equalise pressures between A_1 and A_2
            if not self.pressA_1 == self.pressA_2:
                press1 = self.compute_equalised_press(self.pressA_1, self.volA_1, self.pressA_2, self.volA_2)
                # This should be smooth, 0.7 is a guess, ADJUST
                self.pressA_1 += self.eq_k * (press1 - self.pressA_1)
                self.pressA_2 += self.eq_k * (press1 - self.pressA_2)

            # Equalise temperatures
            w1 = self.volA_1 / (self.volA_1 + self.volA_2)
            w2 = self.volA_2 / (self.volA_1 + self.volA_2)
            temp1 = (w1 * self.tempA_1 + w2 * self.tempA_2)
            # This should be smooth, 0.7 is a guess, ADJUST
            self.tempA_1 += self.eq_k * (temp1 - self.tempA_1)
            self.tempA_2 += self.eq_k * (temp1 - self.tempA_2)

            # Equalise volume of coolant
            if not (self.volA_1 == self.volA_1_default or self.volA_2 == self.volA_2_default):
                v1 = (self.volA_1 + self.volA_2) / 2
                # This should be smooth, 0.7 is a guess, ADJUST
                self.volA_1 += self.eq_k * (v1 - self.volA_1)
                self.volA_2 += self.eq_k * (v1 - self.volA_2)
            
            # Launch or stop pumpA
            if not self.marker_pumpA == self.pumpA:

                if self.pumpA:

                    self.delay_count_pumpA += 1
                    self.flowA += self.pumpA_speed / self.delay_pumpA
                    if self.delay_count_pumpA >= self.delay_pumpA:
                        self.marker_pumpA = True

                elif self.pumpA == False:

                    self.delay_count_pumpA -= 1
                    self.flowA -= self.pumpA_speed / self.delay_pumpA
                    if self.delay_count_pumpA <= 0:
                        self.marker_pumpA = False

        else:

            self.flowA = 0

            # Pump launched, but vlvA_main is closed or meltdown occured
            if not self.marker_pumpA == self.pumpA:

                if self.pumpA:
                    self.delay_count_pumpA += 1
                    self.pressA_1 += 50  # Small pressure change
                    self.pressA_2 -= 50
                    if self.delay_count_pumpA >= self.delay_pumpA:  # Use >= to be safe
                        self.marker_pumpA = True
                else:
                    self.delay_count_pumpA -= 1
                    self.pressA_1 -= 50
                    self.pressA_2 += 50
                    if self.delay_count_pumpA <= 0:  # Use <= to be safe
                        self.marker_pumpA = False

        # Check if there is enough coolant to allow flow
        if self.flowA > 0:
            if self.volA_1 < 0.5 * self.VOLUME_A_1:
                self.flowA = 0  # No need for gradual change since flow stops momentarily if there is not enough coolant

    def update_core(self, power):
        """
        Updates temperature of core, temperature and pressure of A_1

        power - core thermal power, kW
        """

        # Core temperature change
        if self.MELTDOWN == False:
            P_generated = 1000 * power * self.time_step  # Convert kW to W
            heat_coeff_core = self.HEAT_CAPACITY_CORE * self.MASS_CORE
            heat_coeff_A_1 = self.volA_1 * self.WATER_DENSITY * self.HEAT_CAPACITY_WATER
            temp_diff = abs(self.temp_core - self.tempA_1)
            
            if self.flowA > 1.0:

                # Calculate heat transfer using normalized ratio
                normalized_diff = temp_diff / self.temp_core

                # Heat transfer from temperature difference
                P_max_transfer = P_generated * normalized_diff
                
                # Actual heat transfer limited by flow capacity
                P_flow_limit = self.flowA / self.pumpA_speed * P_generated * normalized_diff
                
                # Use the smaller of the two limits
                P_removed = min(P_max_transfer, P_flow_limit)
                
                # Update temperatures
                # Might need adjustment:
                dT_core = (P_generated - P_removed) / heat_coeff_core  # P_generated already multiplied by self.time_step
                heat_loss_coolant = P_removed / heat_coeff_core
                heat_loss_core = (self.temp_core - self.temp_ambient) / (self.temp_core * heat_coeff_core)
                self.temp_core += dT_core - heat_loss_coolant - heat_loss_core

                # Coolant temperature rise
                dT_coolant = P_removed / (self.flowA * self.WATER_DENSITY * self.HEAT_CAPACITY_WATER)
                heat_loss_coolant = (self.tempA_1 - self.temp_ambient) * self.time_step / (self.tempA_1)
                self.tempA_1 += dT_coolant - heat_loss_coolant

            else:

                dT_core = P_generated / heat_coeff_core
                heat_loss_core = 1e-6 * (self.temp_core - self.temp_ambient) * self.time_step
                heat_loss_coolant = 0.2 * P_generated / heat_coeff_core
                self.temp_core += dT_core - heat_loss_core - heat_loss_coolant

                if self.volA_1 > 0.1 * self.VOLUME_A_1:
                    # Slow coolant temperature equalization with ambient
                    heat_gain_coolant = 1e-3 * P_generated / heat_coeff_A_1
                    heat_loss_coolant = 0.001 * (self.tempA_1 - self.temp_ambient) * self.time_step
                    self.tempA_1 += heat_gain_coolant - heat_loss_coolant

        else:
            # Simulate meltdown temp increase
            self.flowA = 0

            dT_core = P_generated * self.time_step / heat_coeff_core
            heat_loss_coolant = 0.05 * P_generated * self.time_step / heat_coeff_core  # 5% of heat goes to coolant
            self.temp_core += dT_core - heat_loss_coolant

            dT_coolant = 1e-3 * P_generated * self.time_step / heat_coeff_A_1  # 5% of heat goes to coolant
            heat_loss_coolant = 1e-3 * (self.tempA_1 - self.temp_ambient) * self.time_step
            self.tempA_1 += dT_coolant - heat_loss_coolant

    def update_B(self):
        """
        Updates temperature and pressure in B_1 and B_2
        """

        flow_factor = (self.flowA / self.MAX_FLOW_A) * (self.flowB / self.MAX_FLOW_B)
        # Need to account for pressure, e.g. minimum heat transfer in case of pressure loss
        press_factor = (self.pressA_2 / self.WORKING_PRESSURE_A) * (self.pressB_1 / self.WORKING_PRESSURE_B)
        dT = flow_factor * press_factor * (self.tempA_2 - self.tempB_1)

        # Simulate heat transfer through steam condenser
        if self.tempB_1 < self.tempA_2:
            self.tempB_1 += 0.8 * dT  # 80% efficiency of intended heat transfer
            self.tempA_2 -= 0.8 * dT
        else:
            self.tempB_1 += 0.2 * dT  # 20 % efficiency of unintended heat transfer
            self.tempA_2 -= 0.2 * dT

        # No flow if either valve closed or pump off
        if not (self.vlvB_main or self.vlvB_2 or self.pumpB):
            self.flowB = 0

        if self.vlvB_main and self.vlvB_2:

            # No need to equalise pressure and temperature since turbine function handles pressure and temperature drop

            # Equalise volume of coolant
            if not (self.volB_1 == self.volB_1_default or self.volB_2 == self.volB_2_default):
                v1 = (self.volB_1 + self.volB_2) / 2
                # This should be smooth, 0.7 is a guess, ADJUST
                self.volB_1 += self.eq_k * (v1 - self.volB_1)
                self.volB_2 += self.eq_k * (v1 - self.volB_2)

            if not self.marker_pumpB == self.pumpB:

                if self.pumpB and self.vlvB_main and self.vlvB_2:

                    self.delay_count_pumpB += 1
                    self.flowB += self.pumpB_speed / self.delay_pumpB  # pumpB gives 15 m^3/s at max power
                    if self.delay_count_pumpB >= self.delay_pumpB:
                        self.marker_pumpB = True

                elif self.pumpB == False:

                    self.delay_count_pumpB -= 1
                    self.flowB -= self.pumpB_speed / self.delay_pumpB
                    if self.delay_count_pumpB <= 0:
                        self.marker_pumpB = False

        else:

            self.flowB = 0

            # Pump launched, but vlvA_main is closed or meltdown occured
            if not self.marker_pumpB == self.pumpB:

                if self.pumpB:
                    self.delay_count_pumpB += 1
                    self.pressB_1 += 50  # Small pressure change
                    self.pressB_2 -= 50
                    if self.delay_count_pumpB >= self.delay_pumpB:  # Use >= to be safe
                        self.marker_pumpB = True
                else:
                    self.delay_count_pumpB -= 1
                    self.pressB_1 -= 10
                    self.pressB_2 += 10
                    if self.delay_count_pumpB <= 0:  # Use <= to be safe
                        self.marker_pumpB = False

        # Check if there is enough coolant to allow flow
        if self.flowB > 0:
            if self.volB_1 < 0.5 * self.VOLUME_B_1:
                self.flowB = 0  # No need for gradual change since flow stops momentarily if there is not enough coolant

    def update_turbine(self):
        """
        Updates turbine power
        """

        if self.pressB_1 > 1000 and self.vlvB_main and self.vlvB_2:

            press_drop = 0.8 * (self.pressB_1 - self.pressB_2)
            new_turbine_power = press_drop * self.flowB
            dP = self.turbine_power - new_turbine_power

            # Using steps of self.launch_speed_turbine or less for smooth launch
            if dP >= 0:
                if dP < self.launch_speed_turbine:  # First check for more common condition for performance
                    increment = dP
                else:
                    increment = self.launch_speed_turbine
            else:
                if dP < self.launch_speed_turbine:
                    increment = -dP
                else:
                    increment = -self.launch_speed_turbine

            self.turbine_power += increment

    def update_reserve(self):
        """
        Updates reserve cooling tank for loop A
        """

        if self.vlvA_4:

            self.flow_from_A_4 = self.MAX_FLOW_A_4 * (1 - self.pressA_2 / self.MAX_PRESSURE)  # Maximum flow when ambient pressure
            dV_out = self.flow_from_A_4 * self.time_step

            if self.vlvA_4:
                self.volA_2 += dV_out
                self.volA_4 -= dV_out

                # Equalise temperatures
                w1 = self.volA_2 / (self.volA_2 + self.volA_4)
                w2 = self.volA_4 / (self.volA_2 + self.volA_4)
                temp1 = (w1 * self.tempA_2 + w2 * self.temp_ambient)  # Reserve water tank has ambient temperature
                self.tempA_2 += self.eq_k * (temp1 - self.tempA_2)
                # No need to calculate temperature in reserve tank

            if self.volA_4 < 0:
                self.volA_4 = 0
                self.flow_4 = 0

            # Simple level change based on flows
            if self.pumpA_4 and self.vlvB_4:

                # Launch or stop pumpA_4
                if not self.marker_pumpA_4 == self.pumpA_4:

                    if self.pumpA_4:
                        delay_count_pumpA_4 += 1

                        # Should create gradual change in flow
                        self.flow_from_A_4 += self.pumpA_4_speed / self.delay_pumpA_4
                        if delay_count_pumpA_4 >= self.delay_pumpA_4:
                            self.marker_pumpA_4 = True

                    elif self.pumpA_4 == False:

                        delay_count_pumpA_4 -= 1
                        self.flowA -= self.pumpA_4_speed / self.delay_pumpA_4
                        if self.delay_count_pumpA <= 0:
                            self.marker_pumpA = False

                dV_in = self.flow_to_A_4 * self.time_step
                if self.volA_4 <= self.VOLUME_A_4 + dV_in and self.pumpA_4:
                    self.flow_to_A_4 = self.pumpA_4_speed
                    self.volA_4 += dV_in  # Reserve tank filled from lake

    def update_relief(self):
        """
        Updates relief tanks for loops A and B
        """

        if self.vlvA_relief:

            dV = self.MAX_RELIEF_FLOW * self.time_step * self.pressA_1 / self.MAX_PRESSURE

            if self.volA_1 > dV and self.volA_2 > dV and self.vol_relief < self.VOLUME_RELIEF:
                self.volA_1 -= dV
                self.vol_relief += dV
            elif self.volA_1 <= dV:
                self.volA_1 = 0
            elif self.volA_2 <= dV:
                self.volA_2 = 0
            elif self.vol_relief >= self.VOLUME_RELIEF:
                self.vol_relief = self.VOLUME_RELIEF

        if self.vlvB_relief:

            dV = self.MAX_RELIEF_FLOW * self.time_step * self.pressB_1 / self.MAX_PRESSURE

            if self.volB_1 > dV and self.volB_2 > dV and self.vol_relief < self.VOLUME_RELIEF:
                self.volA_1 -= dV
                self.vol_relief += dV
            elif self.volA_1 <= dV:
                self.volA_1 = 0
            elif self.volA_2 <= dV:
                self.volA_2 = 0
            elif self.vol_relief >= self.VOLUME_RELIEF:
                self.vol_relief = self.VOLUME_RELIEF

        # Empty relief tank
        if self.vlv_empty_relief:
            self.vol_relief -= self.MAX_RELIEF_FLOW * self.time_step * self.vol_relief / self.VOLUME_RELIEF

    def overpressure_A_1(self):
        """
        Simulates pressure loss for the first part of loop A (core)
        """

        press_coeff = self.pressA_1 / self.MAX_PRESSURE
        dV = 3 * press_coeff * self.time_step  # Maximum flow of around 3 m^3/s

        if self.volA_1 > dV and self.volA_1 > 0:
            self.volA_1 -= dV
        else:
            self.volA_1 = 0

    def overpressure_A_2(self):
        """
        Simulates pressure loss for the second part of loop A (condenser)
        """

        press_coeff = self.pressA_2 / self.MAX_PRESSURE
        dV = 3 * press_coeff * self.time_step  # Maximum flow of around 3 m^3/s

        if self.volA_2 > dV and self.volA_2 > 0:
            self.volA_2 -= dV
        else:
            self.volA_2 = 0

    def overpressure_A_3(self):
        """
        Simulates pressure loss for pressuriser
        """

        press_coeff = self.pressA_2 / self.MAX_PRESSURE
        dV = 3 * press_coeff * self.time_step  # Maximum flow of around 3 m^3/s

        if self.volA_2 > dV and self.volA_3 > 0:
            self.volA_2 -= dV
        else:
            self.volA_2 = 0

    def overpressure_B_1(self):
        """
        Simulates pressure loss for pressuriser
        """

        press_coeff = self.pressB_1 / self.MAX_PRESSURE
        dV = 3 * press_coeff * self.time_step  # Maximum flow of around 3 m^3/s

        if self.volB_1 > dV and self.volB_1 > 0:
            self.volB_1 -= dV
        else:
            self.volB_1 = 0

    def check_catastrophe(self):
        """
        Checks for overpressure, meltdown and other catastrophe conditions
        """

        if not self.catastrophe:

            if self.pressA_1 > self.MAX_PRESSURE:
                self.catastrophe = True
                self.catastrophe_log += "FIRST LOOP A_1 OP\nPRESSURE LOSS\n"
                self.A_OP = True

                self.overpressure_A_1()

            elif self.pressA_2 > self.MAX_PRESSURE:
                self.catastrophe = True
                self.catastrophe_log += "FIRST LOOP A_2 OP\nPRESSURE LOSS\n"
                self.A_OP = True

                self.overpressure_A_2()

            elif self.pressA_3 > self.MAX_PRESSURE:
                self.catastrophe = True
                self.catastrophe_log += "PRESSURISER OP\nPRESSURE LOSS"
                self.A_3_OP = True

                self.overpressure_A_3()

            elif self.pressB_1 > self.MAX_PRESSURE:  # pressB_1 is always >= than pressB_2

                self.catastrophe = True
                self.catastrophe_log += ""
                self.B_OP = True

                self.overpressure_B_1()

            elif self.temp_core > 1100:
                self.catastrophe = True
                self.catastrophe_log += "CORE MELTDOWN\n"
                self.MELTDOWN = True

                self.flowA = 0

        else:

            if self.A_OP:
                self.pressA_2 *= 0.9  # Exponential decay of pressure
                self.volA_2 *= 0.9

            elif self.A_3_OP:

                if self.pressA_3 > self.press_ambient:
                    self.pressA_3 *= 0.9
                
                if self.volA_3:
                    self.volA_3 *= 0.9

            elif self.B_OP:
                
                if self.pressB_1 > self.press_ambient:
                    self.pressB_1 *= 0.9

                self.volB_1 *= 0.9
                self.volB_2 *= 0.9

            elif self.MELTDOWN:

                self.flowA = 0

    def catch_out_of_range(self):
        """
        Manages out of range variables, e.g. self.tempA_1 below self.temp_ambient
        Ideally this should not be needed
        """

        if self.tempA_1 < self.temp_ambient:
            self.tempA_1 = self.temp_ambient

        if self.temp_core < self.temp_ambient:
            self.temp_core = self.temp_ambient

        if self.tempA_2 < self.temp_ambient:
            self.tempA_2 = self.temp_ambient

        if self.tempA_3 < self.temp_ambient:
            self.tempA_3 = self.temp_ambient

        if self.tempB_1 < self.temp_ambient:
            self.tempB_1 = self.temp_ambient

        if self.tempB_2 < self.temp_ambient:
            self.tempB_2 = self.temp_ambient

        if self.pressA_1 < self.press_ambient:
            self.pressA_1 = self.press_ambient

        if self.pressA_2 < self.press_ambient:
            self.pressA_2 = self.press_ambient

        if self.pressA_3 < self.press_ambient:
            self.pressA_3 = self.press_ambient

        if self.pressB_1 < self.press_ambient:
            self.pressB_1 = self.press_ambient

        if self.pressB_2 < self.press_ambient:
            self.pressB_2 = self.press_ambient

        if self.volA_1 > self.VOLUME_A_1:
            self.volA_1 = self.VOLUME_A_1

        if self.volA_2 > self.VOLUME_A_2:
            self.volA_2 = self.VOLUME_A_2

        if self.volA_4 > self.VOLUME_A_4:
            self.volA_4 = self.VOLUME_A_4


def main():

    from time import sleep

    core = ReactorCore()
    sys = ReactorSystems()

    core.initial_power = 10

    sys.vlvA_main = True
    sys.pumpA = True

    time_now = 0

    while True:

        power = core.solve_power(time=time_now)

        sys.simulate_systems(power)

        core.autopilot(500)
        core.check_SCRAM(cond_on=sys.temp_core > 900, cond_off=sys.temp_core < 750)
        print(f"Pow: {power:.3f}")
        print(f"T_core: {sys.temp_core:.3f}")
        print(f"P_A_1: {sys.pressA_1:.3f}")

        time_now += 1

        if sys.tempA_1 < 100:
            sleep(0.01)
        else:
            sleep(0.5)

if __name__ == "__main__":
    main()
