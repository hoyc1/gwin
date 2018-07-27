# Copyright (C) 2018  Charlie Hoy
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""This module provides model classes that assume the noise is Gaussian and
allows for the likelihood to be marginalized over phase and/or time and/or
distance.
"""

import numpy
from scipy import special
from scipy import interpolate

from pycbc import filter
from pycbc.waveform import NoWaveformError
from pycbc.types import Array
from pycbc.filter.matchedfilter import matched_filter_core

from .base_data import BaseDataModel
from .gaussian_noise import GaussianNoise


class MarginalizedGaussianNoise(GaussianNoise):
    r"""The likelihood is analytically marginalized over phase and/or time
    and/or distance.

    For the case of marginalizing over phase, the signal, can be written as:

    .. math::

        \tilde{h}(f; \Theta, \phi) = A(f; \Theta)e^{i\Psi(f; \Theta) + i \phi},

    where :math:`\phi` is an arbitrary phase constant. This phase constant
    can be analytically marginalized over with a uniform prior as follows:
    assuming the noise is stationary and Gaussian (see `GaussianNoise`
    for details), the posterior is:

    .. math::

        p(\Theta,\phi|d)
            &\propto p(\Theta)p(\phi)p(d|\Theta,\phi) \\
            &\propto p(\Theta)\frac{1}{2\pi}\exp\left[
                -\frac{1}{2}\sum_{i}^{N_D} \left<
                    h_i(\Theta,\phi) - d_i, h_i(\Theta,\phi) - d_i
                \right>\right].

    Here, the sum is over the number of detectors :math:`N_D`, :math:`d_i`
    and :math:`h_i` are the data and signal in detector :math:`i`,
    respectively, and we have assumed a uniform prior on :math:`phi \in [0,
    2\pi)`. With the form of the signal model given above, the inner product
    in the exponent can be written as:

    .. math::

        -\frac{1}{2}\left<h_i - d_i, h_i- d_i\right>
            &= \left<h_i, d_i\right> -
               \frac{1}{2}\left<h_i, h_i\right> -
               \frac{1}{2}\left<d_i, d_i\right> \\
            &= \Re\left\{O(h^0_i, d_i)e^{-i\phi}\right\} -
               \frac{1}{2}\left<h^0_i, h^0_i\right> -
               \frac{1}{2}\left<d_i, d_i\right>,

    where:

    .. math::

        h_i^0 &\equiv \tilde{h}_i(f; \Theta, \phi=0); \\
        O(h^0_i, d_i) &\equiv 4 \int_0^\infty
            \frac{\tilde{h}_i^*(f; \Theta,0)\tilde{d}_i(f)}{S_n(f)}\mathrm{d}f.

    Gathering all of the terms that are not dependent on :math:`\phi` together:

    .. math::

        \alpha(\Theta, d) \equiv \exp\left[-\frac{1}{2}\sum_i
            \left<h^0_i, h^0_i\right> + \left<d_i, d_i\right>\right],

    we can marginalize the posterior over :math:`\phi`:

    .. math::

        p(\Theta|d)
            &\propto p(\Theta)\alpha(\Theta,d)\frac{1}{2\pi}
                     \int_{0}^{2\pi}\exp\left[\Re \left\{
                         e^{-i\phi} \sum_i O(h^0_i, d_i)
                     \right\}\right]\mathrm{d}\phi \\
            &\propto p(\Theta)\alpha(\Theta, d)\frac{1}{2\pi}
                     \int_{0}^{2\pi}\exp\left[
                         x(\Theta,d)\cos(\phi) + y(\Theta, d)\sin(\phi)
                     \right]\mathrm{d}\phi.

    The integral in the last line is equal to :math:`2\pi I_0(\sqrt{x^2+y^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind. Thus
    the marginalized log posterior is:

    .. math::

        \log p(\Theta|d) \propto \log p(\Theta) +
            I_0\left(\left|\sum_i O(h^0_i, d_i)\right|\right) -
            \frac{1}{2}\sum_i\left[ \left<h^0_i, h^0_i\right> -
                                    \left<d_i, d_i\right> \right]

    For the case of marginalizing over distance, the signal can be written as,

    .. math::

        \tilde{h}_{j} = \frac{1}{D} \tilde{h}_{j}^{0}

    The distance can be analytically marginalized over with a uniform prior as
    follows:
    assuming the noise is stationary and Gaussian (see `GaussianNoise`
    for details), the likelihood is:

    .. math::

        log L = -\frac{1}{2}\left<d-h|d-h\right>

    We see that :math: `<h|h>` is inversely proportional to distance squared
    and :math: `<h|d>` is inversely proportional to distance. The log
    likelihood is therefore

    .. math::

        \log L = -\frac{1}{2}\left<d|d\right> - \frac{1}{2D^{2}}
                  \left<h|h\right> + \frac{1}{D}\left<h|d\right>

    Consequently, the likelihood marginalised over distance is simply

    .. math::

        \log L = \log\left(\int_{0}^{D}{L p(D) dD}\right)

    If we assume a flat prior

    .. math::

        \log L = \log\left(\int_{0}^{D}{\exp{\log L} dD}\right)

    For the case of marginalizing over time, the signal can be written as,

    .. math::

        \tilde{h}_{j} = \tilde{h}_{j}^{0} \exp\left(-2\pi ij\Delta ft\right)

    The time can be analytically marginalized over with a uniform prior as
    follows:
    assuming the noise is stationary and Gaussian (see `GaussianNoise`
    for details), the likelihood is:

    .. math::

        log L = -\frac{1}{2}\left<d-h|d-h\right>

    We note that :math: `<h|h>` and :math: `<d|d>` are time independent while
    :math: `<d|h>` is dependent of time

    .. math::

        \left<d|h\right>(t) = 4\Delta f\sum_{j=0}^{N/2} \frac{\tilde{d}_{j}^{*}
                              \tilde{h}_{j}^{0}}{S_{j}} \exp\left(-2\pi ij
                              \Delta f t\right)

    For integer timesteps :math: `t=k\Delta t`

    .. math::

        \left<d|h\right>(k\Delta t) = 4\Delta f\sum_{j=0}^{N/2} \frac{
                                      \tilde{d}_{j}^{*}\tilde{h}_{j}^{0}}
                                      {S_{j}}\exp(-2\pi \frac{ijk}{N}

        \left<d|h\right>(k\Delta t) = 2\Delta f\sum_{j=0}^{N} \frac{
                                      \tilde{d}_{j}^{*}\tilde{h}_{j}^{0}}
                                      {S_{j}} \exp(-2\pi \frac{ijk}{N}

    Using a FFT, this expression can be evaluated efficiently for all :math:
    `k`

    .. math::

        \left<d|h\right>(k\Delta t) = 2\Delta f FFT_{k} (\frac{dh}{S})

    since :math:: `\left<h|d\right> = \left<d|h\right>^{*}`,

    .. math::

        \left<d|h\right> + \left<h|d\right> = 4\Delta f FFT_{k} (\frac{dh}{S})

    and so the likelihood marginalised over time is simply

    .. math::

        \log{L} = \log\left(\int_{0}^{T} np.exp(\np.log(L) p(t))\right)

    where p(t) is the prior. If we assume a flat prior then,

    .. math::

        \log{L} = \log\left(\int_{0}^{T} np.exp(\np.log(L))\right)

    Parameters
    ----------
    time_marginalization : bool, optional
        A Boolean operator which determines if the likelihood is marginalized
        over time
    phase_marginalization : bool, optional
        A Boolean operator which determines if the likelihood is marginalized
        over phase
    distance_marginalization : bool, optional
        A Boolean operator which determines if the likelihood is marginalized
        over distance
    **kwargs :
        All other keyword arguments are passed to ``GaussianNoise``.
    """

    def __init__(self, variable_params, data, waveform_generator,
                 f_lower, psds=None, f_upper=None, norm=None,
                 time_marginalization=False, distance_marginalization=False,
                 phase_marginalization=False, **kwargs):
        self._margtime = time_marginalization
        self._margdist = distance_marginalization
        self._margphase = phase_marginalization
        # dictionary containing possible techniques to evalulate the log
        # likelihood ratio.
        loglr_poss = {(1, 1, 1): self._margtimephasedist_loglr,
                      (1, 1, 0): self._margtimedist_loglr,
                      (1, 0, 1): self._margtimephase_loglr,
                      (0, 1, 1): self._margdistphase_loglr,
                      (0, 1, 0): self._margdist_loglr,
                      (1, 0, 0): self._margtime_loglr,
                      (0, 0, 1): self._margphase_loglr}
        # dictionary containing two techniques to calculate the matched
        # filter SNR depending on whether time has been marginalised over or
        # not.
        mfsnr_poss = {(1): self._margtime_mfsnr,
                      (0): self._mfsnr}
        args = (int(self._margtime), int(self._margdist), int(self._margphase))
        self._eval_loglr = loglr_poss[args]
        self._eval_mfsnr = mfsnr_poss[args[0]]
        super(MarginalizedGaussianNoise, self).__init__(variable_params, data,
                                                        waveform_generator,
                                                        f_lower, psds, f_upper,
                                                        norm, **kwargs)

    @property
    def default_stats(self):
        """The stats that ``get_current_stats`` returns by default."""
        if self._margtime:
            return ['logjacobian', 'logprior', 'loglr'] + \
                   ['{}_optimal_snrsq'.format(det) for det in self._data]
        else:
            return ['logjacobian', 'logprior', 'loglr'] + \
                   ['{}_optimal_snrsq'.format(det) for det in self._data] + \
                   ['{}_matchedfilter_snrsq'.format(det) for det in self._data]

    def _margtime_mfsnr(self, template, data):
        """Returns a time series for the matched filter SNR assuming that the
        template and data have both been normalised and whitened.
        """
        snr = matched_filter_core(template, data, h_norm=1, psd=None)
        hd_i = snr[0].numpy().real
        return hd_i

    def _mfsnr(self, template, data):
        """Returns the matched filter SNR assuming that the template and data
        have both been normalised and whitened.
        """
        hd_i = data.inner(template)
        return hd_i

    def _margtimephasedist_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over time, phase and
        istance
        """
        logl = special.logsumexp(numpy.log(special.i0(mf_snr)),
                                 b=self._deltat)
        logl_marg = logl/self._dist_array
        opt_snr_marg = opt_snr/self._dist_array**2
        return special.logsumexp(logl_marg - 0.5*opt_snr_marg, b=self._deltad)

    def _margtimedist_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over time and
        distance
        """
        logl = special.logsumexp(mf_snr, b=self._deltat)
        logl_marg = logl/self._dist_array
        opt_snr_marg = opt_snr/self._dist_array**2
        return special.logsumexp(logl_marg - 0.5*opt_snr_marg, b=self._deltad)

    def _margtimephase_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over time and phase
        """
        return special.logsumexp(numpy.log(special.i0(mf_snr)),
                                 b=self._deltat) - 0.5*opt_snr

    def _margdistphase_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over distance and
        phase
        """
        logl = numpy.log(special.i0(mf_snr))
        logl_marg = logl/self._dist_array
        opt_snr_marg = opt_snr/self._dist_array**2
        return special.logsumexp(logl_marg - 0.5*opt_snr_marg, b=self._deltad)

    def _margdist_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over distance
        """
        mf_snr_marg = mf_snr/self._dist_array
        opt_snr_marg = opt_snr/self._dist_array**2
        return special.logsumexp(mf_snr_marg - 0.5*opt_snr_marg,
                                 b=self._deltad)

    def _margtime_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over time
        """
        return special.logsumexp(mf_snr, b=self._deltat) - 0.5*opt_snr

    def _margphase_loglr(self, mf_snr, opt_snr):
        """Returns the log likelihood ratio marginalized over phase
        """
        return numpy.log(special.i0(mf_snr)) - 0.5*opt_snr

    def _loglr(self):
        r"""Computes the log likelihood ratio,

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio evaluated at the given point.

        """
        params = self.current_params
        try:
            wfs = self._waveform_generator.generate(**params)
        except NoWaveformError:
            return self._nowaveform_loglr()
        opt_snr = 0.
        mf_snr = 0j
        if self._margdist:
            self._priormin = 50.
            self._priormax = 5000.
            self._dist_array = numpy.linspace(self._priormin, self._priormax,
                                              10**4)
            self._deltad = self._dist_array[1] - self._dist_array[0]
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax)
            # time step
            self._deltat = h.delta_t
            if self._kmin >= kmax:
                # if the waveform terminates before the filtering low
                # frequency cutoff, then the loglr is just 0 for this
                # detector
                hh_i = 0.
                hd_i = 0j
            else:
                h[self._kmin:kmax] *= self._weight[det][self._kmin:kmax]
                hh_i = h[self._kmin:kmax].inner(h[self._kmin:kmax]).real
                hd_i = self._eval_mfsnr(h[self._kmin:kmax],
                                        self.data[det][self._kmin:kmax])
            opt_snr += hh_i
            mf_snr += hd_i
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), hh_i)
            if self._margdist or self._margphase:
                setattr(self._current_stats,
                        '{}_matchedfilter_snrsq'.format(det), hd_i)
        mf_snr = abs(mf_snr)
        loglr = self._eval_loglr(mf_snr, opt_snr)
        setattr(self._current_stats, '{}_cplx_loglr'.format(det), loglr)
        return loglr
