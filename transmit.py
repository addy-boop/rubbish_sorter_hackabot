from machine import Pin, SPI
from nrf24l01 import NRF24L01
import time

spi = SPI(0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
csn = Pin(15, mode=Pin.OUT, value=1)
ce = Pin(17, mode=Pin.OUT, value=0)

nrf = NRF24L01(spi, csn, ce, payload_size=1)
nrf.open_tx_pipe(b"\xe1\xf0\xf0\xf0\xf0")
nrf.stop_listening()

led = Pin(25, Pin.OUT)

time.sleep(1)

print("Sending start signal to Pico 1...")
nrf.send(b'S')
led.on()
print("Done - system started")