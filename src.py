import io

import numpy as np
import pandas as pd
import requests

class ApplyActionSpectra:
    """
    Apply CIE sensitivity curves "S", "M", "L", "Rod", and "Mel".
    Accepts a single spectrum (n_wl,) or multiple spectra (n_spectra, n_wl).
    """

    KM = 683.0

    AOPIC_COLS = {
        "S": "s_sc(lambda)",
        "M": "s_mc(lambda)",
        "L": "s_lc(lambda)",
        "Rod": "s_rh(lambda)",
        "Mel": "s_mel(lambda)",
    }

    def __init__(self, irradiance, wavelengths):
        self.irradiance = np.asarray(irradiance, dtype=float)

        if self.irradiance.ndim not in (1, 2):
            raise ValueError("irradiance must be 1D (n_wl,) or 2D (n_spectra, n_wl).")

        self.single = self.irradiance.ndim == 1
        n_wl = self.irradiance.shape[-1]

        if isinstance(wavelengths, tuple) and len(wavelengths) == 2:
            wavelengths = np.linspace(wavelengths[0], wavelengths[1], n_wl)
        elif not isinstance(wavelengths, (list, np.ndarray)):
            raise ValueError(
                "wavelengths must be a tuple (min, max) or a list/array."
            )

        self.wavelengths = np.asarray(wavelengths, dtype=float)

        if self.wavelengths.ndim != 1 or self.wavelengths.shape[0] != n_wl:
            raise ValueError(
                "wavelengths must be 1D with length equal to irradiance.shape[-1]."
            )
        if np.any(np.diff(self.wavelengths) <= 0):
            raise ValueError("wavelengths must be strictly increasing.")

        self.photopic_df = self._read_cie_csv_with_metadata(
            "https://files.cie.co.at/Publications-datasets/CIE_sle_photopic.csv",
            "https://files.cie.co.at/Publications-datasets/CIE_sle_photopic.csv_metadata.json",
        )
        self.aopic_df = self._read_cie_csv_with_metadata(
            "https://files.cie.co.at/Publications-datasets/CIE_a-opic_action_spectra.csv",
            "https://files.cie.co.at/Publications-datasets/CIE_a-opic_action_spectra.csv_metadata.json",
        )

        self._prepare_weights()
        self._integrate()

    @staticmethod
    def _read_cie_csv_with_metadata(csv_url: str, meta_url: str) -> pd.DataFrame:
        meta_resp = requests.get(meta_url)
        meta_resp.raise_for_status()
        titles = [
            c["title"]
            for c in meta_resp.json()["datatableInfo"]["columnHeaders"]
        ]

        csv_resp = requests.get(csv_url)
        csv_resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(csv_resp.content), names=titles, header=0)

        if "lambda" in df.columns:
            df = df.rename(columns={"lambda": "wavelength_nm"})
        else:
            candidates = [
                c for c in df.columns
                if "lambda" in c.lower() or "wavelength" in c.lower()
            ]
            if not candidates:
                raise KeyError("Could not find wavelength column.")
            df = df.rename(columns={candidates[0]: "wavelength_nm"})

        return df

    def _prepare_weights(self):
        wl_user = self.wavelengths

        if "V(lambda)" not in self.photopic_df.columns:
            raise KeyError("Photopic CSV missing 'V(lambda)' column.")

        wl_v = self.photopic_df["wavelength_nm"].to_numpy(dtype=float)
        V = self.photopic_df["V(lambda)"].to_numpy(dtype=float)
        self.V_lambda = np.interp(wl_user, wl_v, V, left=0.0, right=0.0)

        wl_a = self.aopic_df["wavelength_nm"].to_numpy(dtype=float)

        weights = []
        for key, col in self.AOPIC_COLS.items():
            if col not in self.aopic_df.columns:
                raise KeyError(f"Action spectra CSV missing '{col}'.")
            values = self.aopic_df[col].to_numpy(dtype=float)
            w = np.interp(wl_user, wl_a, values, left=0.0, right=0.0)
            setattr(self, f"s_{key}", w)
            weights.append(w)

        # (n_curves, n_wl): order = S, M, L, Rod, Mel
        self.aopic_weights = np.vstack(weights)

    def _integrate(self):
        Ee = np.atleast_2d(self.irradiance)  # (n_spectra, n_wl)
        wl = self.wavelengths

        # (n_spectra,)
        photopic = self.KM * np.trapezoid(Ee * self.V_lambda, wl, axis=-1)

        # (n_spectra, n_curves)
        aopic_irr = np.trapezoid(
            Ee[:, None, :] * self.aopic_weights[None, :, :], wl, axis=-1
        )

        S_irr, M_irr, L_irr, rod_irr, mel_irr = aopic_irr.T

        mp = np.divide(
            self.KM * mel_irr,
            photopic,
            out=np.zeros_like(photopic),
            where=photopic != 0,
        )

        def _sq(x):
            return x[0] if self.single else x

        self.photopic = _sq(photopic)
        self.aopic_irradiance = aopic_irr[0] if self.single else aopic_irr

        self.S_irradiance = _sq(S_irr)
        self.M_irradiance = _sq(M_irr)
        self.L_irradiance = _sq(L_irr)
        self.rod_irradiance = _sq(rod_irr)
        self.mel_irradiance = _sq(mel_irr)

        self.S_lux = self.KM * self.S_irradiance
        self.M_lux = self.KM * self.M_irradiance
        self.L_lux = self.KM * self.L_irradiance
        self.rod_lux = self.KM * self.rod_irradiance
        self.mEDI = self.KM * self.mel_irradiance

        self.MP_ratio = _sq(mp)

    def _out(self, x):
        return float(x) if self.single else np.asarray(x)

    def summary(self) -> dict:
        return {
            "photopic_lux": self._out(self.photopic),
            "mEDI_lux_proxy": self._out(self.mEDI),
            "aopic_lux": {
                "S": self._out(self.S_lux),
                "M": self._out(self.M_lux),
                "L": self._out(self.L_lux),
                "Rod": self._out(self.rod_lux),
                "Mel": self._out(self.mEDI),
            },
            "aopic_irradiance_Wm2": {
                "S": self._out(self.S_irradiance),
                "M": self._out(self.M_irradiance),
                "L": self._out(self.L_irradiance),
                "Rod": self._out(self.rod_irradiance),
                "Mel": self._out(self.mel_irradiance),
            },
            "MP_ratio": self._out(self.MP_ratio),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "photopic_lux": np.atleast_1d(self.photopic),
            "S_lux": np.atleast_1d(self.S_lux),
            "M_lux": np.atleast_1d(self.M_lux),
            "L_lux": np.atleast_1d(self.L_lux),
            "Rod_lux": np.atleast_1d(self.rod_lux),
            "mEDI_lux_proxy": np.atleast_1d(self.mEDI),
            "S_Wm2": np.atleast_1d(self.S_irradiance),
            "M_Wm2": np.atleast_1d(self.M_irradiance),
            "L_Wm2": np.atleast_1d(self.L_irradiance),
            "Rod_Wm2": np.atleast_1d(self.rod_irradiance),
            "Mel_Wm2": np.atleast_1d(self.mel_irradiance),
            "MP_ratio": np.atleast_1d(self.MP_ratio),
        })
