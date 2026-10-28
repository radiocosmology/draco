"""Tests for the TV-channel flagging in draco.analysis.flagging."""

import numpy as np
import pytest

from draco.analysis import flagging
from draco.core import containers

CHIME_FREQ = 800.0 - np.arange(1024) * 400.0 / 1024
# CHORD: 300-1500 MHz, 3200 / 16384 MHz channels
CHORD_FREQ = 300.0 + np.arange(6145) * 3200.0 / 16384


def _tv_channels_flag_old(x, freq, sigma=5, f=0.5):
    """`tv_channels_flag` before frequencies outside the TV band were fixed.

    Started from a flagged fraction of 1, so every frequency outside 398-800 MHz
    was flagged.
    """
    p_false = flagging.sigma_to_p(sigma)
    frac = np.ones_like(x, dtype=np.float32)
    df = np.median(np.abs(np.diff(freq)))
    freq_start = freq - 0.5 * df
    freq_end = freq + 0.5 * df
    for i in range(67):
        fs = 398 + i * 6
        fe = fs + 6
        sel = (freq_end >= fs) & (freq_start <= fe)
        if not sel.any():
            continue
        N = sel.sum()
        k = int(f * N)
        t = flagging.p_to_sigma(flagging.inverse_binom_cdf_prob(k, N, 1 - p_false))
        frac[sel] = (x[sel] > t).mean(axis=0)[np.newaxis, :]
    return frac > f


def _deviations(freq, ntime=100, seed=0, tv=(500.0, 506.0)):
    """Gaussian deviations (in sigma), with excess in one TV channel."""
    rng = np.random.default_rng(seed)
    x = np.abs(rng.standard_normal((len(freq), ntime)))
    in_tv = (freq > tv[0]) & (freq < tv[1])
    x[in_tv, 20:80] += 6.0
    return x, in_tv


def test_chime_band_unchanged():
    """CHIME's band lies inside the TV channels: the mask is as before."""
    for seed in range(3):
        x, _ = _deviations(CHIME_FREQ, seed=seed)
        np.testing.assert_array_equal(
            flagging.tv_channels_flag(x, CHIME_FREQ),
            _tv_channels_flag_old(x, CHIME_FREQ),
        )


def test_outside_tv_band_not_flagged():
    """Frequencies outside 398-800 MHz are never flagged (CHORD's band)."""
    x, in_tv = _deviations(CHORD_FREQ)
    mask = flagging.tv_channels_flag(x, CHORD_FREQ)
    outside = (CHORD_FREQ < 398) | (CHORD_FREQ > 800)
    assert not mask[outside].any()
    # The old code flagged all of them
    assert _tv_channels_flag_old(x, CHORD_FREQ)[outside].all()


def test_tv_channel_flagged():
    """The TV channel with excess is flagged where it has excess, the others are not."""
    x, in_tv = _deviations(CHORD_FREQ)
    mask, frac = flagging.tv_channels_flag(x, CHORD_FREQ, debug=True)
    assert mask[in_tv, 20:80].all()
    assert not mask[in_tv][:, :20].any() and not mask[in_tv][:, 80:].any()
    assert mask[~in_tv].mean() < 0.01
    assert (frac[(CHORD_FREQ < 398) | (CHORD_FREQ > 800)] == 0).all()


def test_band_plan():
    """Other TV channels can be given (base frequency, width, number)."""
    x, in_tv = _deviations(CHORD_FREQ, tv=(1000.5, 1007.5))
    default = flagging.tv_channels_flag(x, CHORD_FREQ)
    custom = flagging.tv_channels_flag(
        x, CHORD_FREQ, base_freq=1000.0, width=8.0, nchannel=4
    )
    assert not default[in_tv].any()
    assert custom[in_tv, 20:80].all()
    assert not custom[(CHORD_FREQ < 1000) | (CHORD_FREQ > 1032)].any()


def _sensitivity(freq, ntime=120, seed=1):
    """A SystemSensitivity container with Gaussian radiometer noise."""
    rng = np.random.default_rng(seed)
    sens = containers.SystemSensitivity(
        freq=freq, pol=np.array(["XX", "YY"]), time=1.7e9 + 10.0 * np.arange(ntime)
    )
    sens.radiometer[:] = 1.0
    sens.measured[:] = 1.0 + 0.01 * rng.standard_normal(sens.measured[:].shape)
    sens.weight[:] = 1.0
    return sens


@pytest.mark.parametrize("tv_fraction", [0.5, None])
def test_sensitivity_mask_wide_band(tv_fraction):
    """RFISensitivityMask on 300-1500 MHz: noise is not flagged, with or without the TV flag."""
    freq = CHORD_FREQ[::8]
    task = flagging.RFISensitivityMask()
    task.read_config(
        {
            "mask_type": "mad",
            "tv_fraction": tv_fraction,
            "base_size": [9, 31],
            "mad_size": [21, 11],
            "win_f_1d": 31,
        }
    )
    assert task.tv_fraction == tv_fraction
    task.setup()
    mask = task.process(_sensitivity(freq)).mask[:]
    assert mask.mean() < 0.02
    assert mask[(freq < 398) | (freq > 800)].mean() < 0.02


def test_tv_fraction_none_config():
    """`tv_fraction: null` in a config gives None (it used to raise)."""
    for cls in (flagging.RFISensitivityMask, flagging.RFIMask):
        task = cls()
        task.read_config({"tv_fraction": None, "stack_ind": 0})
        assert task.tv_fraction is None
        assert not task._tv_mask(
            np.full((10, 4), 100.0), np.linspace(400, 410, 10), 5.0
        ).any()
