# PLACEBÖ – assembly instructions

The hardware edition of PlaceboGPT: a Raspberry Pi Pico, a 58 mm thermal printer and one very large button.
Press it and the world's safest medical advice prints out, identically, every time.

- `placebo-booklet.pdf` – the booklet, 7 pages, A4
- `pages/` – one standalone HTML file per page (the source; edit these)
- `preview/` – PNG render of each page
- `render.js` – rebuilds the PNGs and the PDF from `pages/` (`npm i playwright pdf-lib && npx playwright install chromium && node render.js`)

Firmware (MicroPython for the Pico) is printed on page 6. Art. no. 766.66-0. Hex key included but not required.

Not a medical device. Not medical advice. That is rather the point.
