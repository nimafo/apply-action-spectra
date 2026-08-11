# Apply Action Spectra

Small Python utility for calculating photopic illuminance and CIE S 026 alpha-opic responses from spectral irradiance data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from calibrator import ApplyActionSpectra

wavelengths = np.arange(380, 781, 1)
irradiance = np.ones_like(wavelengths, dtype=float)

result = ApplyActionSpectra(irradiance, wavelengths)

print(result.summary())
print(result.photopic)
print(result.mEDI)
print(result.MP_ratio)
```

`wavelengths` can also be given as `(min_wavelength, max_wavelength)` when the wavelength grid is evenly spaced.

## Outputs

- Photopic illuminance
- S-, M-, L-, rod- and melanopic irradiance
- 683× a-opic lux proxies
- mEDI proxy
- Melanopic/photopic ratio

The class downloads the current CIE action spectra when instantiated.

## References

- CIE photopic spectral luminous efficiency: https://files.cie.co.at/CIE_sle_photopic.csv
- CIE S 026 alpha-opic action spectra: https://files.cie.co.at/CIE_a-opic_action_spectra.csv

## License

MIT
