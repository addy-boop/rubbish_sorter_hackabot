# pico2_body.py
# =============
# Runs on PICO 2 (the "body" node).
# Receives wireless command from Pico 1, runs the belt servo,
# then flips the diverter servo LEFT (can) or RIGHT (paper).
# Shows status on OLED display.
#
# Wiring (nRF24L01+ → Pico 2):
#   VCC  → 3.3V       GND  → GND
#   CE   → GP17       CSN  → GP15
#   SCK  → GP18       MOSI → GP19
#   MISO → GP16       IRQ  → not connected
#
# Wiring (servos via PCA9685):
#   SDA  → GP4        SCL  → GP5
#   VCC  → 3.3V       GND  → GND
#   Belt servo      → PCA9685 channel 0
#   Diverter servo  → PCA9685 channel 1
#
# Wiring (OLED SSD1306 0.96"):
#   SDA  → GP4 (shared I2C bus)
#   SCL  → GP5 (shared I2C bus)
#   VCC  → 3.3V       GND  → GND
#
# Libraries needed (copy to Pico via Thonny):
#   nrf24l01.py  — https://github.com/micropython/micropython-lib
#   pca9685.py   — https://github.com/adafruit/micropython-adafruit-pca9685
#   ssd1306.py   — bundled in MicroPython firmware or install via Thonny

import time
from machine import Pin, SPI, I2C, PWM
from Rubbish_sorter.nrf24l01 import NRF24L01
import ssd1306

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RF_CHANNEL   = 90                        # Must match Pico 1
RX_ADDRESS   = b'\xe1\xf0\xf0\xf0\xf0'  # Must match Pico 1 TX_ADDRESS
PAYLOAD_SIZE = 4

# Belt servo — direct PWM on GP2 (bypasses PCA9685 for simplicity)
BELT_PIN        = 2
BELT_FREQ       = 50     # Hz (standard servo frequency)
BELT_STOP_US    = 1500   # Neutral / stop (for continuous rotation servo)
BELT_SPIN_US    = 1700   # Forward spin speed — tune this
BELT_RUN_TIME   = 1.8    # Seconds to run belt before triggering diverter

# Diverter servo — direct PWM on GP3
DIVERTER_PIN      = 3
DIVERTER_FREQ     = 50
DIVERTER_LEFT_US  = 1000  # CAN bin — tune to your physical setup
DIVERTER_RIGHT_US = 2000  # PAPER bin — tune to your physical setup
DIVERTER_MID_US   = 1500  # Neutral / home position

# OLED
OLED_SDA = 4
OLED_SCL = 5
OLED_W   = 128
OLED_H   = 64
# ───────────────────────────────────────────────────────────────────────────────

led = Pin("LED", Pin.OUT)


def us_to_duty(microseconds, freq=50):
    """Convert microseconds pulse width to 16-bit duty cycle for machine.PWM."""
    period_us = 1_000_000 / freq
    return int((microseconds / period_us) * 65535)


def setup_radio():
    spi = SPI(0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
    csn = Pin(15, mode=Pin.OUT, value=1)
    ce  = Pin(17, mode=Pin.OUT, value=0)
    nrf = NRF24L01(spi, csn, ce, payload_size=PAYLOAD_SIZE)
    nrf.set_channel(RF_CHANNEL)
    nrf.open_rx_pipe(1, RX_ADDRESS)
    nrf.start_listening()
    print("[Radio] Receiver ready on channel", RF_CHANNEL)
    return nrf


def setup_servos():
    belt     = PWM(Pin(BELT_PIN),     freq=BELT_FREQ)
    diverter = PWM(Pin(DIVERTER_PIN), freq=DIVERTER_FREQ)
    # Start with belt stopped, diverter centred
    belt.duty_u16(us_to_duty(BELT_STOP_US))
    diverter.duty_u16(us_to_duty(DIVERTER_MID_US))
    return belt, diverter


def setup_oled():
    i2c = I2C(0, sda=Pin(OLED_SDA), scl=Pin(OLED_SCL), freq=400_000)
    oled = ssd1306.SSD1306_I2C(OLED_W, OLED_H, i2c)
    oled.fill(0)
    oled.text("Rubbish Sorter", 0, 0)
    oled.text("Waiting...", 0, 20)
    oled.show()
    return oled


def oled_update(oled, line1, line2="", line3=""):
    oled.fill(0)
    oled.text("Rubbish Sorter", 0, 0)
    oled.hline(0, 10, 128, 1)
    oled.text(line1, 0, 16)
    oled.text(line2, 0, 30)
    oled.text(line3, 0, 44)
    oled.show()


def sort_item(command, belt, diverter, oled, counts):
    """
    Run the sorting sequence for one item.
    command: 'C' = can, 'P' = paper
    """
    is_can = command == 'C'
    label  = "CAN" if is_can else "PAPER"
    direction = "LEFT" if is_can else "RIGHT"
    diverter_us = DIVERTER_LEFT_US if is_can else DIVERTER_RIGHT_US

    print(f"[Sort] {label} → {direction}")
    oled_update(oled, f"Detected: {label}", f"Sending {direction}", "Belt running...")

    # 1. Run belt
    belt.duty_u16(us_to_duty(BELT_SPIN_US))
    time.sleep(BELT_RUN_TIME)

    # 2. Flip diverter while belt is still running
    diverter.duty_u16(us_to_duty(diverter_us))
    time.sleep(0.6)  # Let item travel over diverter

    # 3. Stop belt
    belt.duty_u16(us_to_duty(BELT_STOP_US))
    time.sleep(0.3)

    # 4. Return diverter to centre
    diverter.duty_u16(us_to_duty(DIVERTER_MID_US))

    # 5. Update counts and display
    counts['cans' if is_can else 'paper'] += 1
    oled_update(oled,
                f"Cans:  {counts['cans']}",
                f"Paper: {counts['paper']}",
                "Ready")

    led.on(); time.sleep(0.15); led.off()
    print(f"[Sort] Done. Cans={counts['cans']} Paper={counts['paper']}")


def main():
    print("[Pico 2] Starting up...")

    nrf          = setup_radio()
    belt, dvrtr  = setup_servos()
    oled         = setup_oled()
    counts       = {'cans': 0, 'paper': 0}

    # Flash LED 3x to signal ready
    for _ in range(3):
        led.on(); time.sleep(0.1); led.off(); time.sleep(0.1)

    oled_update(oled, "Ready!", "Waiting for", "object...")
    print("[Pico 2] Ready — listening for wireless commands")

    while True:
        if nrf.any():
            payload = nrf.recv()
            command = payload.decode().strip()

            if command in ('C', 'P'):
                sort_item(command, belt, dvrtr, oled, counts)
            else:
                print(f"[Radio] Unknown payload: {payload}")

        time.sleep(0.02)


if __name__ == "__main__":
    main()
