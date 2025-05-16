from machine import Pin
import time

# --- Pins Configuration ---
# 74HC595 (Output Shift Register)
data_pin_595 = Pin(13, Pin.OUT)    # DS(14) (Serial Data)
latch_pin_595 = Pin(12, Pin.OUT)   # STCP(12) (Latch)
clock_pin_595 = Pin(14, Pin.OUT)   # SHCP(11) (Clock)

# Initialize pins
latch_pin_595.off()
clock_pin_595.off()
load_pin_165.on()  # Default to shift mode

def write_74hc595(pin_states):
    """Write a list of pin states [QA, QB, ..., QH] to the 74HC595."""
    latch_pin_595.off()  # Prepare to shift data
    
    # Shift out bits (LSB first for 74HC595)
    for i in range(7, -1, -1):  # From QH to QA (MSB first)
        clock_pin_595.off()
        data_pin_595.value(pin_states[i])  # Set QA (index 0) to LSB
        clock_pin_595.on()
        time.sleep_us(5)
    
    latch_pin_595.on()   # Update output registers

count = 0
pins_list = [0 for i in range(8)]
while True:
	count += 1
	if count >= 8:
		count = 0
		pins_list = [0 for i in range(8)]

	pins_list[count] = 1

	write_74hc595(pins_list)
