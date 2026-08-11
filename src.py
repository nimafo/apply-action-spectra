import io

import numpy as np
import pandas as pd
import requests


class ApplyActionSpectra:
    """
    Apply CIE sensitivity curves "S", "M", "L", "Rod", and "Mel".
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

        if isinstance(wavelengths, tuple) and len(wavelengths) == 2:
            wavelengths = np.linspace(
                wavelengths[0], wavelengths[1], len(self.irradiance)
            )
        elif not isinstance(wavelengths, (list, np.ndarray)):
            raise ValueError(
                "wavelengths must be a tuple (min, max) or a list/array."
            )

        self.wavelengths = np.asarray(wavelengths, dtype=float)

        if self.wavelengths.shape != self.irradiance.shape:
            raise ValueError("wavelengths and irradiance must have the same shape.")
        if np.any(np.diff(self.wavelengths) <= 0):
            raise ValueError("wavelengths must be strictly increasing.")

        self.photopic_df = self._read_cie_csv_with_metadata(
            "https://files.cie.co.at/CIE_sle_photopic.csv",
            "https://files.cie.co.at/CIE_sle_photopic.csv_metadata.json",
        )
        self.aopic_df = self._read_cie_csv_with_metadata(
            "https://files.cie.co.at/CIE_a-opic_action_spectra.csv",
            "https://files.cie.co.at/CIE_a-opic_action_spectra.csv_metadata.json",
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

        for key, col in self.AOPIC_COLS.items():
            if col not in self.aopic_df.columns:
                raise KeyError(f"Action spectra CSV missing '{col}'.")
            values = self.aopic_df[col].to_numpy(dtype=float)
            setattr(
                self,
                f"s_{key}",
                np.interp(wl_user, wl_a, values, left=0.0, right=0.0),
            )

    @staticmethod
    def _trapz_nm(y, x_nm):
        return np.trapezoid(y, x_nm)

    def _integrate(self):
        Ee = self.irradiance
        wl = self.wavelengths

        self.photopic = self.KM * self._trapz_nm(Ee * self.V_lambda, wl)

        self.S_irradiance = self._trapz_nm(Ee * self.s_S, wl)
        self.M_irradiance = self._trapz_nm(Ee * self.s_M, wl)
        self.L_irradiance = self._trapz_nm(Ee * self.s_L, wl)
        self.rod_irradiance = self._trapz_nm(Ee * self.s_Rod, wl)
        self.mel_irradiance = self._trapz_nm(Ee * self.s_Mel, wl)

        self.S_lux = self.KM * self.S_irradiance
        self.M_lux = self.KM * self.M_irradiance
        self.L_lux = self.KM * self.L_irradiance
        self.rod_lux = self.KM * self.rod_irradiance
        self.mEDI = self.KM * self.mel_irradiance

        self.MP_ratio = self.mEDI / self.photopic if self.photopic else 0.0

    def summary(self) -> dict:
        return {
            "photopic_lux": float(self.photopic),
            "mEDI_lux_proxy": float(self.mEDI),
            "aopic_lux": {
                "S": float(self.S_lux),
                "M": float(self.M_lux),
                "L": float(self.L_lux),
                "Rod": float(self.rod_lux),
                "Mel": float(self.mEDI),
            },
            "aopic_irradiance_Wm2": {
                "S": float(self.S_irradiance),
                "M": float(self.M_irradiance),
                "L": float(self.L_irradiance),
                "Rod": float(self.rod_irradiance),
                "Mel": float(self.mel_irradiance),
            },
            "MP_ratio": float(self.MP_ratio),
        }
