# pico1_brain.py
# ==============
# Runs on PICO 1 (the "brain" node).
# Receives 'C' or 'P' over USB serial from the laptop detector script,
# then forwards the command wirelessly to Pico 2 via nRF24L01+.
#
# Wiring (nRF24L01+ → Pico 1):
#   VCC  → 3.3V
#   GND  → GND
#   CE   → GP17
#   CSN  → GP15
#   SCK  → GP18
#   MOSI → GP19
#   MISO → GP16
#   IRQ  → not connected
#
# Install nrf24l01 library:
#   Copy nrf24l01.py from https://github.com/micropython/micropython-lib
#   to your Pico via Thonny or mpremote

import sys
import time
from machine import Pin, SPI
from Rubbish_sorter.nrf24l01 import NRF24L01

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# RF channel and addresses — must match pico2_body.py exactly
RF_CHANNEL    = 90
TX_ADDRESS    = b'\xe1\xf0\xf0\xf0\xf0'
PAYLOAD_SIZE  = 4   # bytes
# ───────────────────────────────────────────────────────────────────────────────

# Status LED (onboard LED)
led = Pin("LED", Pin.OUT)

def setup_radio():
    spi = SPI(0,
              sck=Pin(18),
              mosi=Pin(19),
              miso=Pin(16))
    csn = Pin(15, mode=Pin.OUT, value=1)
    ce  = Pin(17, mode=Pin.OUT, value=0)

    nrf = NRF24L01(spi, csn, ce, payload_size=PAYLOAD_SIZE)
    nrf.set_channel(RF_CHANNEL)
    nrf.open_tx_pipe(TX_ADDRESS)
    nrf.stop_listening()
    print("[Radio] Transmitter ready on channel", RF_CHANNEL)
    return nrf


def send_command(nrf, command):
    """
    Send a 4-byte padded command string wirelessly.
    command should be 'C' (can) or 'P' (paper).
    """
    payload = command.encode().ljust(PAYLOAD_SIZE)  # Pad to PAYLOAD_SIZE bytes
    try:
        nrf.send(payload)
        print(f"[Radio] Sent: {command}")
        # Blink LED to confirm send
        led.on()
        time.sleep(0.1)
        led.off()
        return True
    except OSError:
        print("[Radio] Send failed — is Pico 2 powered on?")
        return False


def main():
    print("[Pico 1] Starting up...")
    nrf = setup_radio()
    print("[Pico 1] Waiting for commands from laptop via USB serial...")

    # Flash LED 3x to signal ready
    for _ in range(3):
        led.on(); time.sleep(0.1); led.off(); time.sleep(0.1)

    while True:
        # Read one byte at a time from USB serial (laptop sends 'C' or 'P')
        if sys.stdin in __import__('select').select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)

            if char == 'C':
                print("[Serial] Received: CAN")
                send_command(nrf, 'C')

            elif char == 'P':
                print("[Serial] Received: PAPER")
                send_command(nrf, 'P')

            else:
                print(f"[Serial] Unknown command: '{char}' — ignored")

        time.sleep(0.01)  # Small yield to avoid busy-looping


if __name__ == "__main__":
    main()
